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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- 하이퍼파라미터 관리: Config 클래스 ---
class Config:
    seed = 42
    use_topN = 30 # Morgan TopN 피처 개수
    batch_size = 64
    learning_rate = 0.001
    patience = 20 # Early Stopping을 위한 patience
    epochs = 100 # 최대 에포크 수

    # GNN 모델 관련 파라미터
    node_feat_dim = 9 # atom_features의 반환값 개수 (atomic_num, total_hs 등 9가지)
    edge_feat_dim = 9 # get_bond_features의 반환값 개수 (bond_type, conjugated, aromatic, stereo_onehot(5), in_ring)

    # Global feature 관련 (RDKit Descriptors 20개 + MACCSKeys 167개 = 187개)
    global_feat_base_dim = 20 + 167

    # 최종 global_feat_dim은 global_feat_base_dim + use_topN으로 결정됨

# 설정
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
try:
    train_df = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
except FileNotFoundError as e:
    print(f"오류: 필요한 CSV 파일이 없습니다. 경로를 확인해주세요: {e}")
    exit() # 파일이 없으면 스크립트 종료

smiles_train = train_df['Canonical_Smiles'].tolist()
y = train_df['Inhibition'].values
smiles_test = test_df['Canonical_Smiles'].tolist()

# --- Morgan FP ---
def get_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 경고: 유효하지 않은 SMILES (Morgan FP)에 대한 로그 출력
        # print(f"경고: 유효하지 않은 SMILES (Morgan FP) 감지 - {smiles}")
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        # 경고: Morgan FP 생성 중 오류 발생에 대한 로그 출력
        print(f"경고: Morgan FP 생성 중 오류 발생 ({smiles}): {e}")
        return None

print("Morgan Fingerprint 생성 중...")
morgan_train_raw = [get_morgan_fp(s) for s in tqdm(smiles_train, desc="Train Morgan FP")]
morgan_test_raw = [get_morgan_fp(s) for s in tqdm(smiles_test, desc="Test Morgan FP")]

# None 값 필터링 및 유효 인덱스 추출
valid_train_idx = [i for i, fp in enumerate(morgan_train_raw) if fp is not None]
smiles_train = [smiles_train[i] for i in valid_train_idx]
y = y[valid_train_idx]
morgan_train = np.array([morgan_train_raw[i] for i in valid_train_idx])

valid_test_smiles_original_idx = [i for i, fp in enumerate(morgan_test_raw) if fp is not None]
smiles_test = [smiles_test[i] for i in valid_test_smiles_original_idx]
morgan_test = np.array([morgan_test_raw[i] for i in valid_test_smiles_original_idx])
# 참고: test_df의 원래 순서를 유지하기 위해 valid_test_smiles_original_idx를 저장해야 함 (제출 시 필요)

print(f"유효한 Morgan Train 데이터셋 크기: {len(morgan_train)}")
print(f"유효한 Morgan Test 데이터셋 크기: {len(morgan_test)}")

# --- Morgan TopN (Config.use_topN 사용) ---
top_idx_path = f'./_save/morgan_top{Config.use_topN}_from_combined.npy'
if not os.path.exists(top_idx_path):
    print(f"경고: {top_idx_path} 파일이 없습니다. Morgan TopN 피처를 사용할 수 없습니다.")
    # 대체 로직 또는 에러 처리
    # 여기서는 간단히 전체 Morgan FP를 사용하는 것으로 대체 (성능 저하 가능성 있음)
    print("Morgan TopN 대신 전체 Morgan FP를 사용합니다.")
    morgan_train_top = morgan_train
    morgan_test_top = morgan_test
    # Config.use_topN을 전체 Morgan FP 길이로 업데이트
    Config.use_topN = morgan_train.shape[1] if morgan_train.shape[0] > 0 else 0
else:
    top_idx = np.load(top_idx_path)
    morgan_train_top = morgan_train[:, top_idx]
    morgan_test_top = morgan_test[:, top_idx]
print(f"Morgan Top-{Config.use_topN} 피처 사용.")

# --- RDKit + MACCS ---
rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, Descriptors.RingCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge
]

def get_rdkit_and_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 경고: 유효하지 않은 SMILES (RDKit/MACCS)에 대한 로그 출력
        # print(f"경고: 유효하지 않은 SMILES (RDKit/MACCS) 감지 - {smiles}")
        return None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc, nan=0.0).astype(np.float32) # NaN을 0.0으로 처리
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return np.concatenate([desc, maccs_arr.astype(np.float32)])
    except Exception as e:
        # 경고: RDKit/MACCS 피처 생성 중 오류 발생에 대한 로그 출력
        print(f"경고: RDKit/MACCS 피처 생성 중 오류 발생 ({smiles}): {e}")
        return None

# --- Node Feature ---
def atom_features(atom):
    return [
        atom.GetAtomicNum(), atom.GetTotalNumHs(), int(atom.GetHybridization()),
        atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
        atom.GetDegree(), atom.GetExplicitValence(), atom.GetImplicitValence(),
    ]

# --- Stereo One-hot ---
def stereo_onehot(bond):
    onehot = [0] * 5
    stereo = int(bond.GetStereo())
    if 0 <= stereo < 5:
        onehot[stereo] = 1
    return onehot

# --- Edge Feature (Config.edge_feat_dim 사용) ---
def get_bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.GetIsAromatic(),
        *stereo_onehot(bond), # Stereo one-hot 벡터가 여기에 언팩됨 (5개)
        bond.IsInRing(),
    ] # 총 1 + 1 + 1 + 5 + 1 = 9개 피처

# --- Graph 변환 ---
def smiles_to_graph_with_global(smiles, global_feat, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 경고: 그래프 변환 중 유효하지 않은 SMILES에 대한 로그 출력
        # print(f"경고: 그래프 변환 실패 - 유효하지 않은 SMILES: {smiles}")
        return None

    try:
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = get_bond_features(bond)
            edge_indices += [[i, j], [j, i]] # 양방향 엣지 추가
            edge_attrs += [feat, feat]

        if not edge_indices: # 엣지가 없는 경우 (단일 원자 분자 등)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, Config.edge_feat_dim), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).view(-1)
        y_tensor = torch.tensor([label], dtype=torch.float) if label is not None else None

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat_tensor, y=y_tensor)
    except Exception as e:
        # 경고: 그래프 변환 중 예상치 못한 오류 발생에 대한 로그 출력
        print(f"경고: 그래프 변환 중 오류 발생 ({smiles}): {e}")
        return None

# --- 통합 피처 구성 및 Graph 데이터 생성 ---
print("\n그래프 데이터셋 생성 중...")
X_train_processed, X_test_processed = [], []

# Train 데이터 처리
for i, s in enumerate(tqdm(smiles_train, desc="Processing Train Data")):
    rdkit_maccs = get_rdkit_and_maccs(s)
    if rdkit_maccs is None:
        continue
    global_feat = np.concatenate([rdkit_maccs, morgan_train_top[i]])
    if global_feat.shape[0] != Config.global_feat_base_dim + Config.use_topN:
        print(f"오류: 트레인 데이터 {s}의 글로벌 피처 차원 불일치. 예상: {Config.global_feat_base_dim + Config.use_topN}, 실제: {global_feat.shape[0]}")
        continue
    X_train_processed.append((s, global_feat, y[i]))

# Test 데이터 처리 (원본 인덱스를 유지하기 위해 별도 처리)
test_smiles_to_process = []
test_original_indices = []
for i, s_original in enumerate(test_df['Canonical_Smiles'].tolist()):
    # 현재 smiles_test는 유효한 Morgan FP를 가진 SMILES만 포함
    # 따라서, test_df의 모든 SMILES를 다시 확인하여 일치하는 경우에만 처리
    if s_original in smiles_test: # 유효성 검사된 smiles_test에 포함된 경우만 처리
        test_smiles_to_process.append(s_original)
        test_original_indices.append(i) # 원본 DataFrame에서의 인덱스 저장

# morgan_test는 이미 유효한 SMILES 순서로 되어 있음
current_morgan_test_idx = 0
for i, s in enumerate(tqdm(test_smiles_to_process, desc="Processing Test Data")):
    rdkit_maccs = get_rdkit_and_maccs(s)
    if rdkit_maccs is None:
        continue # 이 SMILES에 대해 RDKit/MACCS 피처를 생성할 수 없으면 건너뜀
    
    if current_morgan_test_idx >= len(morgan_test_top):
        print(f"오류: Test Morgan FP 인덱스 초과. SMILES: {s}")
        continue

    global_feat = np.concatenate([rdkit_maccs, morgan_test_top[current_morgan_test_idx]])
    if global_feat.shape[0] != Config.global_feat_base_dim + Config.use_topN:
        print(f"오류: 테스트 데이터 {s}의 글로벌 피처 차원 불일치. 예상: {Config.global_feat_base_dim + Config.use_topN}, 실제: {global_feat.shape[0]}")
        continue
    X_test_processed.append((s, global_feat, test_original_indices[i])) # 원본 인덱스 저장
    current_morgan_test_idx += 1


train_data = [smiles_to_graph_with_global(s, f, l) for s, f, l in X_train_processed]
test_data_with_indices = []
for s, f, original_idx in X_test_processed:
    graph_data = smiles_to_graph_with_global(s, f)
    if graph_data is not None:
        test_data_with_indices.append((graph_data, original_idx)) # (그래프 데이터, 원본 인덱스) 튜플 저장

train_data = [d for d in train_data if d is not None]
test_data = [d[0] for d in test_data_with_indices] # 그래프 데이터만 추출
test_submission_original_indices = [d[1] for d in test_data_with_indices] # 제출을 위한 원본 인덱스 저장

print(f"최종 Train Graph 데이터 개수: {len(train_data)}")
print(f"최종 Test Graph 데이터 개수: {len(test_data)}")


# --- GNN 모델 정의 (Config.node_feat_dim, Config.global_feat_dim, Config.edge_feat_dim 사용) ---
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=Config.node_feat_dim,
                 global_feat_dim=Config.global_feat_base_dim + Config.use_topN,
                 edge_feat_dim=Config.edge_feat_dim):
        super().__init__()
        # GATv2Conv는 edge_dim 인자를 통해 edge feature를 고려할 수 있음
        self.gat1 = GATv2Conv(node_feat_dim, 64, heads=4, dropout=0.2, edge_dim=edge_feat_dim)
        self.bn1 = BatchNorm1d(64 * 4) # heads * out_channels
        self.gat2 = GATv2Conv(64 * 4, 128, heads=4, dropout=0.2, edge_dim=edge_feat_dim)
        self.bn2 = BatchNorm1d(128 * 4)
        self.relu = ReLU()
        self.dropout = Dropout(0.3)
        self.fc1 = Linear(128 * 4 + global_feat_dim, 128) # GNN 출력 + 글로벌 피처
        self.fc2 = Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_feat = data.global_feat.to(x.device)
        edge_attr = edge_attr.to(x.device) # edge_attr도 디바이스로 이동

        # GATv2 Conv Layers
        x = self.relu(self.bn1(self.gat1(x, edge_index, edge_attr)))
        x = self.relu(self.bn2(self.gat2(x, edge_index, edge_attr)))

        # Graph Pooling (노드 임베딩을 그래프 임베딩으로 변환)
        x = global_mean_pool(x, batch) # 각 그래프별 평균 풀링

        # Global Features 결합
        # global_feat의 차원이 batch_size * global_feat_dim이 되도록 조정
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            # global_feat가 (N_graphs * F_global) 형태가 아닌 경우 처리
            # 예를 들어, DataLoader에서 Batch 객체가 아닌 Data 객체 하나만 들어올 때
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0) # (F_global,) -> (1, F_global)
            # 배치 내 각 그래프의 global_feat가 하나의 텐서로 합쳐졌을 때
            # 이를 각 그래프에 맞게 분리해야 함
            # PyG DataLoader가 제대로 batching하면 보통 이 부분 필요 없음.
            # 하지만 안전을 위해 size(0) 체크
            if global_feat.size(0) != x.size(0):
                 # 에러 메시지 또는 디버깅 로깅 추가
                raise ValueError(f"Global feature batch size mismatch: expected {x.size(0)}, got {global_feat.size(0)}")

        x = torch.cat([x, global_feat], dim=1) # GNN 출력과 글로벌 피처 결합

        # MLP Layers
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze() # 최종 예측값 (회귀이므로 squeeze)

# --- 학습 준비 ---
train_idx, val_idx = train_test_split(range(len(train_data)), test_size=0.2, random_state=Config.seed)
train_loader = DataLoader([train_data[i] for i in train_idx], batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader([train_data[i] for i in val_idx], batch_size=Config.batch_size)
test_loader = DataLoader(test_data, batch_size=Config.batch_size) # test_data는 그래프 데이터만 포함

# --- 학습 ---
model = GATv2WithGlobal().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
loss_fn = torch.nn.SmoothL1Loss() # Huber Loss (MSE와 MAE의 장점 결합)

best_score, best_rmse, best_corr = -np.inf, 0, 0
counter, patience = 0, Config.patience
model_save_path = f'./_save/gnn_edge{Config.edge_feat_dim}_stereo1hot_morgan{Config.use_topN}.pt'

print("\n모델 학습 시작...")
for epoch in range(1, Config.epochs + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())

    preds, trues = np.concatenate(preds), np.concatenate(trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    corr = pearsonr(trues, preds)[0]

    # Dacon 평가 지표 계산: 0.5 * (1 - RMSE_normalized) + 0.5 * CORR_clipped
    y_range = trues.max() - trues.min()
    normalized_rmse = rmse / y_range if y_range > 0 else 1.0 # 0으로 나누는 경우 방지
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

    print(f"--- [Epoch {epoch}/{Config.epochs}] Train Loss: {total_loss/len(train_loader):.4f} | Val RMSE: {rmse:.4f}, Val Corr: {corr:.4f}, Val Score: {score:.4f} ---")

    if score > best_score:
        best_score, best_rmse, best_corr = score, rmse, corr
        counter = 0 # 성능 개선 시 카운터 초기화
        torch.save(model.state_dict(), model_save_path)
        print(f"새로운 Best Score 달성! 모델 저장: {model_save_path}")
    else:
        counter += 1
        print(f"성능 미개선. Early Stopping 카운터: {counter}/{Config.patience}")
        if counter >= patience:
            print(f"Early Stopping! {Config.patience} 에포크 동안 성능 개선이 없었습니다.")
            break

# --- 테스트 예측 ---
print("\n테스트 예측 시작...")
model.load_state_dict(torch.load(model_save_path)) # Best 모델 로드
model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting Test Data"):
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())

final_test_predictions = np.concatenate(test_preds)

# submit_df에 예측값 할당 (test_submission_original_indices를 사용하여 원래 순서대로 채우기)
# 먼저 submit_df의 Inhibition 컬럼을 -1 등으로 초기화하여 예측하지 못한 값들을 구분
submit_df['Inhibition'] = np.nan # 기본값으로 NaN 설정
for i, pred_val in zip(test_submission_original_indices, final_test_predictions):
    submit_df.loc[i, 'Inhibition'] = pred_val

# 예측되지 않은 값 (예: 그래프 변환에 실패한 SMILES)이 있다면, submit_df에 NaN으로 남아있을 수 있음
# 이런 경우를 어떻게 처리할지 결정해야 함 (예: 0으로 채우기, 평균값으로 채우기 등)
# 여기서는 NaN으로 그대로 두거나, 0으로 채우는 예시
submit_df['Inhibition'].fillna(0, inplace=True) # 예측 실패한 값은 0으로 채움 (또는 test 데이터의 평균 등)

submission_file_name = f'./_save/gnn_edge{Config.edge_feat_dim}_stereo1hot_morgan{Config.use_topN}_submission.csv'
submit_df.to_csv(submission_file_name, index=False)

# --- 결과 출력 ---
print("\n--- [최종 결과 - 저장된 Best Model 기준] ---")
print(f"Best Validation Score : {best_score:.4f}")
print(f"Best Validation RMSE  : {best_rmse:.4f}")
print(f"Best Validation Corr  : {best_corr:.4f}")
print(f"✅ 제출 파일 저장 완료: {submission_file_name}")