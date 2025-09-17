import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit import DataStructs
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# Optuna 시각화를 위한 라이브러리 추가
import optuna.visualization
import plotly.io as pio # 이미지 저장을 위해 필요할 수 있습니다.

# --- 하이퍼파라미터 관리: Config 클래스 (원래 위치) ---
class Config:
    seed = 42
    use_topN = 30 # Morgan TopN 피처 개수 (만약 파일 없으면 전체 Morgan FP 사용)
    batch_size = 64
    epochs = 100 # 각 Optuna trial 내 K-Fold 학습의 최대 에포크 수 및 최종 모델 학습의 최대 에포크 수
    patience = 7 # Early Stopping patience

    # GNN 모델 관련 파라미터
    node_feat_dim = 14 # 14개 피처 유지
    edge_feat_dim = 9
    
    # Global feature 관련
    # 초기에는 임시 값을 설정 (나중에 실제 rdkit_desc_list 길이로 업데이트될 것임)
    rdkit_desc_dim = 20 # 이 값은 아래에서 실제 길이로 업데이트됩니다.
    maccs_keys_dim = 167
    
    # K-Fold 설정
    n_splits = 3 # K-Fold 분할 개수를 3으로 조정 (튜닝 시간을 고려하여 유지)

# 설정
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 한글 폰트 설정 (필요에 따라 주석 처리 또는 수정)
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (이전과 동일)
try:
    # 데이터 경로를 실제 환경에 맞게 조정해주세요.
    train_df = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
except FileNotFoundError as e:
    print(f"오류: 필요한 CSV 파일이 없습니다. 경로를 확인해주세요: {e}")
    exit()

smiles_train_raw_initial = train_df['Canonical_Smiles'].tolist()
y_original_initial = train_df['Inhibition'].values # 원본 스케일의 Inhibition 값
y_transformed_initial = y_original_initial # log1p 변환 제거 유지

smiles_test_raw_initial = test_df['Canonical_Smiles'].tolist()

# --- Morgan FP --- (이전과 동일)
def get_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        return None

print("Morgan Fingerprint 생성 중 (훈련 데이터)...")
morgan_train_raw = [get_morgan_fp(s) for s in tqdm(smiles_train_raw_initial, desc="Train Morgan FP")]

# 유효한 Morgan FP를 가진 훈련 데이터만 필터링
valid_train_indices_morgan = [i for i, fp in enumerate(morgan_train_raw) if fp is not None]
smiles_train_filtered_by_morgan = [smiles_train_raw_initial[i] for i in valid_train_indices_morgan]
morgan_train = np.array([morgan_train_raw[i] for i in valid_train_indices_morgan])
y_transformed_filtered_for_graph = y_transformed_initial[valid_train_indices_morgan]


print("Morgan Fingerprint 생성 중 (테스트 데이터)...")
morgan_test_raw = [get_morgan_fp(s) for s in tqdm(smiles_test_raw_initial, desc="Test Morgan FP")]

valid_test_indices_morgan = [i for i, fp in enumerate(morgan_test_raw) if fp is not None]
smiles_test_filtered_by_morgan = [smiles_test_raw_initial[i] for i in valid_test_indices_morgan]
morgan_test = np.array([morgan_test_raw[i] for i in valid_test_indices_morgan])

# 원본 test_df 인덱스를 유지하기 위해 필터링된 테스트 데이터의 원본 인덱스 저장
test_original_indices_filtered = [test_df.index[i] for i in valid_test_indices_morgan]

print(f"유효한 Morgan Train 데이터셋 크기: {len(morgan_train)}")
print(f"유효한 Morgan Test 데이터셋 크기: {len(morgan_test)}")

# Morgan TopN 피처 로드 (파일이 없으면 전체 Morgan FP 사용)
top_idx_path = f'./_save/morgan_top{Config.use_topN}_from_combined.npy'
if not os.path.exists(top_idx_path):
    print(f"경고: {top_idx_path} 파일이 없습니다. Morgan TopN 피처를 사용할 수 없습니다.")
    print("Morgan TopN 대신 전체 Morgan FP를 사용합니다. 이 경우 Config.use_topN이 변경됩니다.")
    if morgan_train.shape[0] > 0:
        morgan_train_top = morgan_train
        morgan_test_top = morgan_test
        Config.use_topN = morgan_train.shape[1] # 실제 사용될 Morgan FP 차원
    else: # morgan_train 자체가 비어있는 경우
        morgan_train_top = np.array([])
        morgan_test_top = np.array([])
        Config.use_topN = 0
else:
    top_idx = np.load(top_idx_path)
    if len(top_idx) == 0: # top_idx가 비어있는 경우 (예: 피처 중요도 분석 시 모든 피처가 중요하지 않다고 판단)
        print(f"경고: {top_idx_path} 파일에 유효한 TopN 인덱스가 없습니다. Morgan FP를 사용하지 않습니다.")
        morgan_train_top = np.array([])
        morgan_test_top = np.array([])
        Config.use_topN = 0
    else:
        Config.use_topN = len(top_idx) 
        morgan_train_top = morgan_train[:, top_idx]
        morgan_test_top = morgan_test[:, top_idx]
print(f"Morgan 피처 차원: {Config.use_topN}")


# --- RDKit Descriptors 목록 정의 ---
# Descriptors.NumRings 제거 및 Config.rdkit_desc_dim 업데이트 로직 반영
rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge
]

# rdkit_desc_list가 정의된 후, Config 클래스의 rdkit_desc_dim을 업데이트
Config.rdkit_desc_dim = len(rdkit_desc_list)
print(f"RDKit Descriptors 차원 (업데이트): {Config.rdkit_desc_dim}")


def get_rdkit_and_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc, nan=0.0).astype(np.float32)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((Config.maccs_keys_dim,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return desc, maccs_arr.astype(np.float32)
    except Exception as e:
        return None, None

# --- Node Feature (14개 피처) ---
def atom_features(atom):
    return [
        atom.GetAtomicNum(),  # 원자 번호
        atom.GetTotalNumHs(),  # 수소 원자 총 개수
        int(atom.GetHybridization()),  # 혼성 궤도 (sp, sp2, sp3 등)
        atom.GetFormalCharge(),  # 형식 전하
        int(atom.GetIsAromatic()),  # 방향족 여부
        int(atom.IsInRing()),  # 고리 내 포함 여부
        atom.GetDegree(),  # 원자의 차수 (연결된 비수소 원자 수)
        atom.GetExplicitValence(),  # 명시적 원자가
        atom.GetImplicitValence(),  # 묵시적 원자가
        atom.GetMass(), # 원자 질량
        atom.GetNumRadicalElectrons(), # 라디칼 전자 수
        atom.GetTotalValence(), # 총 원자가
        atom.GetChiralTag(), # 카이랄 태그 (정수 값)
        atom.GetIsotope(), # 동위원소 (정수 값)
    ]

# --- Stereo One-hot ---
def stereo_onehot(bond):
    onehot = [0] * 5
    stereo = int(bond.GetStereo())
    if 0 <= stereo < 5:
        onehot[stereo] = 1
    return onehot

# --- Edge Feature ---
def get_bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.GetIsAromatic(),
        *stereo_onehot(bond),
        bond.IsInRing(),
    ]

# --- Graph 변환 함수 (original_y_idx 추가) ---
def smiles_to_graph_with_global(smiles, global_feat, label=None, original_y_idx=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = get_bond_features(bond)
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [feat, feat]

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, Config.edge_feat_dim), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).unsqueeze(0)
        y_tensor = torch.tensor([label], dtype=torch.float) if label is not None else None

        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat_tensor, y=y_tensor)
        data_obj.idx_in_original_y = original_y_idx # 원본 y 배열에서의 인덱스 저장
        return data_obj
    except Exception as e:
        return None

# --- 통합 피처 구성 및 Graph 데이터 생성 ---
print("\n그래프 데이터셋 생성 중...")

# Train 데이터의 RDKit Descriptors와 MACCS Keys 추출
train_rdkit_descs_raw, train_maccs_keys_raw = [], []
graph_creation_valid_indices_in_filtered_smiles = [] 

for i, s in enumerate(tqdm(smiles_train_filtered_by_morgan, desc="Extracting Train RDKit/MACCS & Graph")):
    rdkit_desc, maccs_keys = get_rdkit_and_maccs(s)
    
    if rdkit_desc is None or len(rdkit_desc) != Config.rdkit_desc_dim or maccs_keys is None or len(maccs_keys) != Config.maccs_keys_dim:
        continue

    if i >= len(morgan_train_top) or morgan_train_top[i] is None or (Config.use_topN > 0 and morgan_train_top[i].size == 0):
        continue

    train_rdkit_descs_raw.append(rdkit_desc)
    train_maccs_keys_raw.append(maccs_keys)
    graph_creation_valid_indices_in_filtered_smiles.append(i)

morgan_train_top_final = np.array([morgan_train_top[idx] for idx in graph_creation_valid_indices_in_filtered_smiles])
smiles_train_final = [smiles_train_filtered_by_morgan[idx] for idx in graph_creation_valid_indices_in_filtered_smiles]
y_transformed_final = np.array([y_transformed_filtered_for_graph[idx] for idx in graph_creation_valid_indices_in_filtered_smiles])
y_original_indices_for_graph_final = [valid_train_indices_morgan[idx] for idx in graph_creation_valid_indices_in_filtered_smiles] 


# Test 데이터의 RDKit Descriptors와 MACCS Keys 추출
test_rdkit_descs_raw, test_maccs_keys_raw = [], []
graph_creation_valid_indices_in_test_filtered_smiles = [] 

for i, s in enumerate(tqdm(smiles_test_filtered_by_morgan, desc="Extracting Test RDKit/MACCS & Graph")):
    rdkit_desc, maccs_keys = get_rdkit_and_maccs(s)
    
    if rdkit_desc is None or len(rdkit_desc) != Config.rdkit_desc_dim or maccs_keys is None or len(maccs_keys) != Config.maccs_keys_dim:
        continue

    if i >= len(morgan_test_top) or morgan_test_top[i] is None or (Config.use_topN > 0 and morgan_test_top[i].size == 0):
        continue

    test_rdkit_descs_raw.append(rdkit_desc)
    test_maccs_keys_raw.append(maccs_keys)
    graph_creation_valid_indices_in_test_filtered_smiles.append(i)

morgan_test_top_final = np.array([morgan_test_top[idx] for idx in graph_creation_valid_indices_in_test_filtered_smiles])
smiles_test_final = [smiles_test_filtered_by_morgan[idx] for idx in graph_creation_valid_indices_in_test_filtered_smiles]
test_original_indices_final = [test_original_indices_filtered[idx] for idx in graph_creation_valid_indices_in_test_filtered_smiles]


# RDKit Descriptors만 StandardScaler 적용 (데이터가 비어있지 않은 경우에만)
train_rdkit_descs_np = np.array(train_rdkit_descs_raw)
test_rdkit_descs_np = np.array(test_rdkit_descs_raw)

if train_rdkit_descs_np.shape[0] > 0:
    scaler = StandardScaler()
    train_rdkit_descs_scaled = scaler.fit_transform(train_rdkit_descs_np)
    test_rdkit_descs_scaled = scaler.transform(test_rdkit_descs_np)
else:
    train_rdkit_descs_scaled = np.array([]) 
    test_rdkit_descs_scaled = np.array([])


# 최종 Graph Data 객체 생성 (이전과 동일)
train_data = []
for i in tqdm(range(len(smiles_train_final)), desc="Creating Train Graphs"):
    components = []
    if train_rdkit_descs_scaled.shape[0] > 0 and i < train_rdkit_descs_scaled.shape[0]:
        components.append(train_rdkit_descs_scaled[i])
    if train_maccs_keys_raw and i < len(train_maccs_keys_raw):
        components.append(train_maccs_keys_raw[i])
    if morgan_train_top_final.shape[0] > 0 and i < morgan_train_top_final.shape[0]:
        components.append(morgan_train_top_final[i])
    
    if not components:
        continue

    global_feat_combined = np.concatenate(components)

    data_obj = smiles_to_graph_with_global(
        smiles_train_final[i], 
        global_feat_combined, 
        y_transformed_final[i], 
        original_y_idx=y_original_indices_for_graph_final[i]
    )
    
    if data_obj is not None:
        train_data.append(data_obj)

test_data_with_indices = []
for i in tqdm(range(len(smiles_test_final)), desc="Creating Test Graphs"):
    components = []
    if test_rdkit_descs_scaled.shape[0] > 0 and i < test_rdkit_descs_scaled.shape[0]:
        components.append(test_rdkit_descs_scaled[i])
    if test_maccs_keys_raw and i < len(test_maccs_keys_raw):
        components.append(test_maccs_keys_raw[i])
    if morgan_test_top_final.shape[0] > 0 and i < morgan_test_top_final.shape[0]:
        components.append(morgan_test_top_final[i])

    if not components:
        continue

    global_feat_combined = np.concatenate(components)

    graph_data = smiles_to_graph_with_global(smiles_test_final[i], global_feat_combined)
    if graph_data is not None:
        test_data_with_indices.append((graph_data, test_original_indices_final[i]))

test_data = [d[0] for d in test_data_with_indices]
test_submission_original_indices = [d[1] for d in test_data_with_indices]

print(f"최종 Train Graph 데이터 개수: {len(train_data)}")
print(f"최종 Test Graph 데이터 개수: {len(test_data)}")

# --- GNN 모델 정의 (Optuna로부터 파라미터 받도록 수정) ---
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim, global_feat_dim, edge_feat_dim,
                 gat_out_channels_1, gat_out_channels_2, num_heads, gat_dropout,
                 fc_hidden_dim, fc_dropout):
        super().__init__()
        self.global_feat_dim = max(0, global_feat_dim)

        self.gat1 = GATv2Conv(node_feat_dim, gat_out_channels_1, heads=num_heads, dropout=gat_dropout, edge_dim=edge_feat_dim)
        self.bn1 = BatchNorm1d(gat_out_channels_1 * num_heads)
        
        self.gat2 = GATv2Conv(gat_out_channels_1 * num_heads, gat_out_channels_2, heads=num_heads, dropout=gat_dropout, edge_dim=edge_feat_dim)
        self.bn2 = BatchNorm1d(gat_out_channels_2 * num_heads)
        self.relu = ReLU()
        
        # 잔차 연결을 위한 투영 레이어 추가
        self.res_proj1 = Linear(node_feat_dim, gat_out_channels_1 * num_heads)
        self.res_proj2 = Linear(gat_out_channels_1 * num_heads, gat_out_channels_2 * num_heads)
        
        fc1_input_dim = gat_out_channels_2 * num_heads + self.global_feat_dim
        self.fc1 = Linear(fc1_input_dim, fc_hidden_dim)
        self.dropout_fc = Dropout(fc_dropout)
        self.fc2 = Linear(fc_hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if self.global_feat_dim > 0:
            global_feat = data.global_feat.to(x.device)
        else:
            global_feat = torch.empty((x.shape[0] if batch is None else batch.max().item() + 1, 0), device=x.device)
            
        edge_attr = edge_attr.to(x.device)

        # 첫 번째 GAT 층에 잔차 연결 적용
        x_skip1 = x 
        x = self.relu(self.bn1(self.gat1(x_skip1, edge_index, edge_attr))) # GATv2Conv의 입력은 x_skip1로
        x = x + self.res_proj1(x_skip1) # 투영된 입력 + 출력

        # 두 번째 GAT 층에 잔차 연결 적용
        x_skip2 = x 
        x = self.relu(self.bn2(self.gat2(x_skip2, edge_index, edge_attr))) # GATv2Conv의 입력은 x_skip2로
        x = x + self.res_proj2(x_skip2) 

        x = global_mean_pool(x, batch)
        
        if self.global_feat_dim > 0:
            x = torch.cat([x, global_feat], dim=1)

        x = self.dropout_fc(self.relu(self.fc1(x)))
        return self.fc2(x)


# --- Optuna Objective 함수 ---
def objective(trial):
    # 튜닝할 하이퍼파라미터 제안
    gat_out_channels_1 = trial.suggest_categorical('gat_out_channels_1', [64, 128, 256])
    gat_out_channels_2 = trial.suggest_categorical('gat_out_channels_2', [128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    gat_dropout = trial.suggest_uniform('gat_dropout', 0.1, 0.4)

    fc_hidden_dim = trial.suggest_categorical('fc_hidden_dim', [64, 128, 256])
    fc_dropout = trial.suggest_uniform('fc_dropout', 0.2, 0.5)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    # K-Fold 교차 검증 시작
    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    fold_val_scores = []
    
    global_feat_dim_actual = Config.rdkit_desc_dim + Config.maccs_keys_dim + Config.use_topN

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_data)):
        
        fold_train_data = [train_data[i] for i in train_indices]
        fold_val_data = [train_data[i] for i in val_indices]
        
        fold_val_y_original_indices = [d.idx_in_original_y for d in fold_val_data]
        fold_val_y_original = y_original_initial[fold_val_y_original_indices] 

        fold_train_loader = DataLoader(fold_train_data, batch_size=Config.batch_size, shuffle=True)
        fold_val_loader = DataLoader(fold_val_data, batch_size=Config.batch_size)

        model = GATv2WithGlobal(
            node_feat_dim=Config.node_feat_dim,
            global_feat_dim=global_feat_dim_actual,
            edge_feat_dim=Config.edge_feat_dim,
            gat_out_channels_1=gat_out_channels_1,
            gat_out_channels_2=gat_out_channels_2,
            num_heads=num_heads,
            gat_dropout=gat_dropout,
            fc_hidden_dim=fc_hidden_dim,
            fc_dropout=fc_dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.SmoothL1Loss()

        best_fold_score = -np.inf
        fold_counter = 0

        for epoch_in_fold in range(1, Config.epochs + 1):
            model.train()
            for batch in fold_train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = loss_fn(out, batch.y.unsqueeze(1))
                loss.backward()
                optimizer.step()

            model.eval()
            preds_list_transformed = []
            with torch.no_grad():
                for batch in fold_val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    preds_list_transformed.append(out.cpu().numpy())

            preds_transformed = np.concatenate(preds_list_transformed).squeeze() 
            preds_original_scale = preds_transformed 

            rmse = np.sqrt(mean_squared_error(fold_val_y_original, preds_original_scale))
            corr = pearsonr(fold_val_y_original, preds_original_scale)[0]
            
            y_range = fold_val_y_original.max() - fold_val_y_original.min()
            normalized_rmse = rmse / y_range if y_range > 0 else 1.0
            score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

            # Optuna Pruner에 보고
            current_optuna_step_for_pruning = (fold * Config.epochs) + epoch_in_fold
            
            trial.report(score, current_optuna_step_for_pruning)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if score > best_fold_score:
                best_fold_score = score
                fold_counter = 0
            else:
                fold_counter += 1
                if fold_counter >= Config.patience:
                    break # Early Stopping

        fold_val_scores.append(best_fold_score)

    mean_val_score = np.mean(fold_val_scores)
    return mean_val_score

# --- Optuna Study 생성 및 실행 ---
print("\nOptuna 하이퍼파라미터 튜닝 시작...")
sampler = TPESampler(seed=Config.seed)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)

# study_name 변경: 잔차 연결 추가를 명시
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="gatv2_kfold_tuning_extended_node_feat_es_final_k3_n100_lr_scheduler_res_conn") 
# ⭐ 여기에서 n_trials 값을 조정하세요 (예: 200) ⭐
study.optimize(objective, n_trials=300, show_progress_bar=True)

print("\n--- Optuna 튜닝 결과 ---")
print("Best trial:")
trial = study.best_trial

print(f"   Value (Mean K-Fold Validation Score): {trial.value:.4f}")
print("   Params: ")
for key, value in trial.params.items():
    print(f"     {key}: {value}")

# ⭐⭐⭐ Optuna 시각화 코드 시작 ⭐⭐⭐
print("\n--- Optuna 시각화 ---")

# 최적화 이력 시각화 (성능 개선 추이)
fig_history = optuna.visualization.plot_optimization_history(study)
fig_history.show()
# fig_history.write_image("./_save/optuna_optimization_history.png") # 이미지 파일로 저장하려면 이 주석을 해제하세요.

# 하이퍼파라미터 중요도 시각화
fig_importance = optuna.visualization.plot_param_importances(study)
fig_importance.show()
# fig_importance.write_image("./_save/optuna_param_importances.png")

# 각 하이퍼파라미터와 성능 간의 관계 시각화 (슬라이스 플롯)
fig_slice = optuna.visualization.plot_slice(study)
fig_slice.show()
# fig_slice.write_image("./_save/optuna_slice_plot.png")

# (선택 사항) 평행 좌표 플롯: 여러 하이퍼파라미터 조합과 성능을 동시에 시각화
# fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
# fig_parallel.show()
# fig_parallel.write_image("./_save/optuna_parallel_coordinate.png")
# ⭐⭐⭐ Optuna 시각화 코드 끝 ⭐⭐⭐

# --- 최적의 파라미터로 최종 모델 학습 및 예측 ---
print("\n최적의 파라미터로 최종 모델 학습 및 테스트 예측 시작...")

test_loader = DataLoader(test_data, batch_size=Config.batch_size)

# 최종 학습을 위한 훈련/검증 데이터 분할
train_indices_for_final_split, val_indices_for_final_split = train_test_split(
    list(range(len(train_data))), # <-- 괄호 수정된 부분
    test_size=0.1, # 10%를 최종 검증 세트로 사용
    random_state=Config.seed,
    shuffle=True
)

train_data_final_split = [train_data[i] for i in train_indices_for_final_split]
val_data_final_split = [train_data[i] for i in val_indices_for_final_split]

val_y_original_indices_final = [d.idx_in_original_y for d in val_data_final_split]
val_y_original_final = y_original_initial[val_y_original_indices_final]

full_train_loader = DataLoader(train_data_final_split, batch_size=Config.batch_size, shuffle=True)
final_val_loader = DataLoader(val_data_final_split, batch_size=Config.batch_size)

final_global_feat_dim = Config.rdkit_desc_dim + Config.maccs_keys_dim + Config.use_topN
final_model = GATv2WithGlobal(
    node_feat_dim=Config.node_feat_dim,
    global_feat_dim=final_global_feat_dim,
    edge_feat_dim=Config.edge_feat_dim,
    gat_out_channels_1=trial.params['gat_out_channels_1'],
    gat_out_channels_2=trial.params['gat_out_channels_2'],
    num_heads=trial.params['num_heads'],
    gat_dropout=trial.params['gat_dropout'],
    fc_hidden_dim=trial.params['fc_hidden_dim'],
    fc_dropout=trial.params['fc_dropout']
).to(device)

final_optimizer = torch.optim.Adam(final_model.parameters(), lr=trial.params['learning_rate'])
final_loss_fn = torch.nn.SmoothL1Loss()

# 학습률 스케줄러 정의: ReduceLROnPlateau
final_scheduler = ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# 모델 저장 파일명 변경: 잔차 연결 추가를 명시
final_model_save_path = './_save/final_best_gatv2_model_from_optuna_es_14feat_k3_n100_lr_scheduler_res_conn.pt' 
final_best_val_score = -np.inf
final_patience_counter = 0

print("Final model training with Early Stopping and Learning Rate Scheduler...")
for epoch in range(1, Config.epochs + 1):
    final_model.train()
    total_train_loss = 0
    for batch in tqdm(full_train_loader, desc=f"Final Train Epoch {epoch}"):
        batch = batch.to(device)
        final_optimizer.zero_grad()
        out = final_model(batch)
        loss = final_loss_fn(out, batch.y.unsqueeze(1))
        loss.backward()
        final_optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(full_train_loader)

    final_model.eval()
    val_preds_list = []
    total_val_loss = 0
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            out = final_model(batch)
            loss = final_loss_fn(out, batch.y.unsqueeze(1))
            total_val_loss += loss.item()
            val_preds_list.append(out.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(final_val_loader)
    val_preds_original_scale = np.concatenate(val_preds_list).squeeze()

    val_rmse = np.sqrt(mean_squared_error(val_y_original_final, val_preds_original_scale))
    val_corr = pearsonr(val_y_original_final, val_preds_original_scale)[0]

    val_y_range = val_y_original_final.max() - val_y_original_final.min()
    val_normalized_rmse = val_rmse / val_y_range if val_y_range > 0 else 1.0
    current_val_score = 0.5 * (1 - min(val_normalized_rmse, 1)) + 0.5 * np.clip(val_corr, 0, 1)

    print(f"--- [Final Epoch {epoch}/{Config.epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Score: {current_val_score:.4f} ---")

    # 학습률 스케줄러 업데이트
    final_scheduler.step(current_val_score)

    if current_val_score > final_best_val_score:
        final_best_val_score = current_val_score
        torch.save(final_model.state_dict(), final_model_save_path)
        print(f"✅ 최적 모델 저장 완료: {final_model_save_path} (Val Score: {final_best_val_score:.4f})")
        final_patience_counter = 0
    else:
        final_patience_counter += 1
        if final_patience_counter >= Config.patience:
            print(f"🚫 Early Stopping! Validation score did not improve for {Config.patience} epochs.")
            break

# --- 테스트 예측 ---
print("\n--- 테스트 예측 시작 ---")
final_model.load_state_dict(torch.load(final_model_save_path)) # 최적 모델 로드
final_model.eval()
test_preds_transformed = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting Test Data"):
        batch = batch.to(device)
        out = final_model(batch)
        test_preds_transformed.append(out.cpu().numpy())

final_test_predictions = np.concatenate(test_preds_transformed).squeeze()

submit_df['Inhibition'] = np.nan
for i, pred_val in zip(test_submission_original_indices, final_test_predictions):
    submit_df.loc[i, 'Inhibition'] = max(0, pred_val)

submit_df['Inhibition'] = submit_df['Inhibition'].fillna(0)

# 제출 파일명 변경: 잔차 연결 추가를 명시
submission_file_name = f'./_save/gnn_gatv2_optuna_kfold_submission_es_14feat_k3_n100_lr_scheduler_res_conn.csv' 
submit_df.to_csv(submission_file_name, index=False)

# --- 최종 결과 출력 ---
print("\n--- [최종 결과 - Optuna 튜닝 완료] ---")
print(f"Optuna Best Trial Mean K-Fold Validation Score : {trial.value:.4f}")
print(f"Final Model Best Validation Score (with ES): {final_best_val_score:.4f}")
print(f"✅ 제출 파일 저장 완료: {submission_file_name}")