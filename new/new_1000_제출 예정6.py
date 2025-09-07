import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.AllChem import rdmolops
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import optuna

# ----------------------------------------------------
#               설정 및 데이터 전처리
# ----------------------------------------------------

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
try:
    train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
except FileNotFoundError:
    print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

smiles_train = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values
smiles_test = test['Canonical_Smiles'].tolist()

# SMARTS 및 노드 피처 정의
smart_patterns_raw = ['[C]=[O]', '[NH2]', 'c1ccccc1']
smarts_patterns = [Chem.MolFromSmarts(p) for p in smart_patterns_raw if Chem.MolFromSmarts(p)]
electronegativity = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66, 14: 1.90}
covalent_radius = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39, 14: 1.11}

rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, Descriptors.RingCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge
]

def atom_features(atom):
    num = atom.GetAtomicNum()
    eneg = electronegativity.get(num, 0)
    radius = covalent_radius.get(num, 0)
    smarts_match = sum(atom.GetOwningMol().HasSubstructMatch(p) for p in smarts_patterns)
    return [
        num, eneg, radius,
        atom.GetTotalNumHs(), int(atom.GetHybridization()), atom.GetFormalCharge(),
        int(atom.GetIsAromatic()), int(atom.IsInRing()), smarts_match
    ]

# ----------------------------------------------------
#               엣지(결합) 피처 정의 (인접 원자 정보 추가)
# ----------------------------------------------------
def bond_features(bond):
    bt = bond.GetBondType()
    
    # 기본 결합 특성
    features = [
        int(bt == Chem.BondType.SINGLE), # 단일 결합
        int(bt == Chem.BondType.DOUBLE), # 이중 결합
        int(bt == Chem.BondType.TRIPLE), # 삼중 결합
        int(bt == Chem.BondType.AROMATIC), # 방향족 결합
        int(bond.GetIsConjugated()), # 공액 여부
        int(bond.IsInRing()), # 고리 내 결합 여부
        int(bond.GetStereo() == Chem.BondStereo.STEREOANY), # 스테레오화학 정보
        int(bond.GetStereo() == Chem.BondStereo.STEREOCIS),
        int(bond.GetStereo() == Chem.BondStereo.STEREOTRANS)
    ]
    
    # 추가: 인접 원자의 일부 정보
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    # 원자 번호
    features.append(begin_atom.GetAtomicNum())
    features.append(end_atom.GetAtomicNum())
    
    # 전기음성도
    features.append(electronegativity.get(begin_atom.GetAtomicNum(), 0))
    features.append(electronegativity.get(end_atom.GetAtomicNum(), 0))
    
    # 형식 전하
    features.append(begin_atom.GetFormalCharge())
    features.append(end_atom.GetFormalCharge())
    
    # 혼성화 상태 (정수형)
    features.append(int(begin_atom.GetHybridization()))
    features.append(int(end_atom.GetHybridization()))

    return features

# 엣지 피처의 차원 (bond_features 함수의 반환 리스트 길이)
# 'CC'는 유효한 단일 결합을 가진 간단한 분자이므로, 이를 이용해 차원 계산
try:
    EDGE_FEATURE_DIM = len(bond_features(Chem.MolFromSmiles('CC').GetBonds()[0]))
except Exception:
    # 만약 'CC'가 오류를 내면, 기본값으로 설정하거나 다른 간단한 분자 시도
    # 실제 환경에서 bond_features 함수가 모든 필요한 케이스에 대해 올바른 길이를 반환하는지 확인 필요
    print("WARNING: Could not determine EDGE_FEATURE_DIM automatically. Using a default value. Please verify.")
    EDGE_FEATURE_DIM = 17 # 이전 9개 + 원자번호2 + 전기음도2 + 형식전하2 + 혼성화2 = 17 예상 (확인 필요)

print(f"새롭게 정의된 EDGE_FEATURE_DIM: {EDGE_FEATURE_DIM}")

# ----------------------------------------------------
#               MACCS 키 중요도 분석 (추가된 부분)
# ----------------------------------------------------

def get_only_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return maccs_arr.astype(np.float32)
    except Exception:
        return None

# 모든 MACCS 키 데이터 추출
print("MACCS 키 중요도 분석을 위한 데이터 추출 중...")
maccs_data_list = []
valid_y = []
valid_smiles = []

for i, s in enumerate(tqdm(smiles_train, desc="Extracting all MACCS keys")):
    maccs_feat = get_only_maccs(s)
    if maccs_feat is not None:
        maccs_data_list.append(maccs_feat)
        valid_y.append(y[i])
        valid_smiles.append(s)

maccs_data = np.array(maccs_data_list)
valid_y = np.array(valid_y)

if maccs_data.shape[0] == 0:
    print("유효한 MACCS 키 데이터가 없습니다. 종료합니다.")
    exit()

# RandomForestRegressor를 이용한 특징 중요도 계산
print(f"랜덤 포레스트 모델 학습 중... (MACCS 데이터 shape: {maccs_data.shape})")
rf_model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
rf_model.fit(maccs_data, valid_y)

feature_importances = rf_model.feature_importances_
maccs_importance_dict = {i: importance for i, importance in enumerate(feature_importances)}
sorted_maccs_by_importance = sorted(maccs_importance_dict.items(), key=lambda item: item[1], reverse=True)

# 상위 30개 MACCS 키 인덱스 선택
top_30_maccs_indices = [idx for idx, _ in sorted_maccs_by_importance[:30]]
top_30_maccs_indices.sort() # 인덱스 정렬
print(f"\n✅ 랜덤 포레스트 중요도로 선택된 Top 30 MACCS Key 인덱스 (0-indexed): {top_30_maccs_indices}")
# ----------------------------------------------------
#               전처리 함수 재정의 및 데이터셋 생성
# ----------------------------------------------------

def get_rdkit_and_selected_maccs(smiles, selected_maccs_indices):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc).astype(np.float32)
        
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

        # 선택된 MACCS 키만 추출
        selected_maccs_arr = maccs_arr[selected_maccs_indices].astype(np.float32)
        
        return np.concatenate([desc, selected_maccs_arr])
    except Exception:
        return None

# `smiles_to_graph_final` 함수 수정: edge_attr 추가
def smiles_to_graph_final(smiles, label=None, selected_maccs_indices=None, scaler=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    global_feat = get_rdkit_and_selected_maccs(smiles, selected_maccs_indices)
    if global_feat is None: return None
    
    # 정규화 적용
    global_feat_scaled = scaler.transform(global_feat.reshape(1, -1)).flatten()
    
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    
    # 엣지 인덱스 및 엣지 피처 생성
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        # 양방향 엣지 추가
        edge_indices.append([u, v])
        edge_indices.append([v, u])
        # 엣지 피처 추가 (양방향 동일)
        bf = bond_features(bond)
        edge_attrs.append(bf)
        edge_attrs.append(bf)
    
    if not edge_indices: # 분자에 결합이 없는 경우 (예: 단일 원자)
        # PyTorch Geometric Data 객체는 비어있는 엣지_index와 엣지_attr을 처리할 수 있어야 함
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) # 엣지 피처 텐서

    global_feat_tensor = torch.tensor(global_feat_scaled, dtype=torch.float).view(-1)
    
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat_tensor, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, edge_attr=edge_attr)

# 정규화
global_features_all = [
    get_rdkit_and_selected_maccs(s, top_30_maccs_indices) 
    for s in smiles_train
    if get_rdkit_and_selected_maccs(s, top_30_maccs_indices) is not None
]
global_features_all = np.array(global_features_all)
scaler = StandardScaler()
scaler.fit(global_features_all)

# 전처리
print("\n최종 데이터셋 생성 중... (선택된 MACCS 키만 사용)")
data_list = [smiles_to_graph_final(s, l, top_30_maccs_indices, scaler) for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train), desc="Processing train data")]
data_list = [d for d in data_list if d is not None]

test_data = [smiles_to_graph_final(s, None, top_30_maccs_indices, scaler) for s in tqdm(smiles_test, desc="Processing test data")]
test_data = [d for d in test_data if d is not None]

# ----------------------------------------------------
#               모델 정의 (글로벌 피처 차원 수정 및 엣지 피처 추가)
# ----------------------------------------------------
# RDKit 디스크립터 20개 + 선택된 MACCS 키 30개 = 50
global_feat_dim_updated = 20 + len(top_30_maccs_indices)

class GATv2WithGlobal(Module):
    # edge_dim 파라미터 추가
    def __init__(self, node_feat_dim=9, edge_feat_dim=EDGE_FEATURE_DIM, global_feat_dim=global_feat_dim_updated, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        # GATv2Conv에 edge_dim 추가
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.bn2 = BatchNorm1d(hidden_channels * heads)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

    def forward(self, data):
        # edge_attr도 data에서 추출
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        global_feat = data.global_feat.to(x.device)
        
        # GATv2Conv에 edge_attr 전달
        x = self.relu(self.bn1(self.gat1(x, edge_index, edge_attr)))
        x = self.relu(self.bn2(self.gat2(x, edge_index, edge_attr)))
        
        x = global_mean_pool(x, batch)
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            global_feat = global_feat.view(x.size(0), -1)
        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ----------------------------------------------------
#               Optuna Objective 함수
# ----------------------------------------------------
# (suggest_loguniform -> suggest_float(log=True)로 변경)

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])

    model = GATv2WithGlobal(
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        edge_feat_dim=EDGE_FEATURE_DIM # 추가된 부분: edge_feat_dim 전달
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=batch_size)

    best_score = -np.inf
    patience_counter = 0
    patience = 10

    for epoch in range(1, 101):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                preds.append(out.cpu().numpy())
                trues.append(batch.y.view(-1).cpu().numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        corr = pearsonr(trues, preds)[0]
        y_range = trues.max() - trues.min()
        normalized_rmse = rmse / y_range if y_range > 0 else 1.0
        score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_score

# ----------------------------------------------------
#               Optuna Study 실행
# ----------------------------------------------------

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n\n--- Optuna 최적화 결과 ---")
print("최적 하이퍼파라미터:", study.best_params)
print("최고 Score:", study.best_value)

# ----------------------------------------------------
#               최적 하이퍼파라미터로 최종 학습
# ----------------------------------------------------

best_params = study.best_params

print("\n--- 최적의 하이퍼파라미터로 최종 모델 학습 시작 ---")
model = GATv2WithGlobal(
    hidden_channels=best_params['hidden_channels'],
    heads=best_params['heads'],
    dropout=best_params['dropout'],
    edge_feat_dim=EDGE_FEATURE_DIM # 추가된 부분: edge_feat_dim 전달
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn = torch.nn.SmoothL1Loss()

# 훈련-검증 데이터 분할
train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
final_train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=best_params['batch_size'], shuffle=True)
final_val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=best_params['batch_size'])
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

# 최종 학습 루프
best_val_score = -np.inf
patience_counter = 0
patience = 20 # 더 긴 조기 종료 인내심

for epoch in range(1, 101):
    model.train()
    for batch in final_train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

    # 검증 단계
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            out = model(batch)
            val_preds.append(out.cpu().numpy())
            val_trues.append(batch.y.view(-1).cpu().numpy())

    val_preds, val_trues = np.concatenate(val_preds), np.concatenate(val_trues)
    rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
    corr = pearsonr(val_trues, val_preds)[0]
    std = np.std(val_trues)
    current_val_score = 0.5 * (1 - min(rmse / std, 1)) + 0.5 * corr

    print(f"Epoch {epoch:03d} | Validation Score: {current_val_score:.4f} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")

    # 조기 종료 및 모델 저장
    if current_val_score > best_val_score:
        best_val_score = current_val_score
        patience_counter = 0
        torch.save(model.state_dict(), f'./_save/gnn_rf_top30_best_model.pt')
        print(f"새로운 최고 점수 달성, 모델 저장 완료: {best_val_score:.4f}")
    else:
        patience_counter += 1
        print(f"Score가 개선되지 않았습니다. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"조기 종료! (Epoch: {epoch})")
            break

# ----------------------------------------------------
#               최종 예측 및 저장
# ----------------------------------------------------

# 가장 성능이 좋았던 모델을 불러와서 예측
final_model_path = './_save/gnn_rf_top30_best_model.pt'
model.load_state_dict(torch.load(final_model_path))
model.eval()

test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())

submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_rf_top30_submission.csv', index=False)

print("\n✅ 최종 예측 완료: ./_save/gnn_rf_top30_submission.csv")
print(f"  사용된 MACCS 키 개수: {len(top_30_maccs_indices)}")
print(f"  최종 global_feat 차원: {global_feat_dim_updated}")
print(f"  엣지 피처 차원: {EDGE_FEATURE_DIM}")