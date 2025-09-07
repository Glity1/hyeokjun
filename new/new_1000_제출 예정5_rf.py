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
#               MACCS 키 중요도 분석 (추가된 부분)
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
#               전처리 함수 재정의 및 데이터셋 생성
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

def smiles_to_graph_with_global_selected(smiles, label=None, selected_maccs_indices=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    global_feat = get_rdkit_and_selected_maccs(smiles, selected_maccs_indices)
    if global_feat is None: return None
    
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = (torch.tensor(adj) > 0).nonzero(as_tuple=False).t().contiguous()
    global_feat = torch.tensor(global_feat, dtype=torch.float).view(-1)
    
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat)
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat)

# 정규화
global_features_all = [
    get_rdkit_and_selected_maccs(s, top_30_maccs_indices) 
    for s in smiles_train
    if get_rdkit_and_selected_maccs(s, top_30_maccs_indices) is not None
]
global_features_all = np.array(global_features_all)
scaler = StandardScaler()
scaler.fit(global_features_all)

def smiles_to_graph_final(smiles, label=None, selected_maccs_indices=None, scaler=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    global_feat = get_rdkit_and_selected_maccs(smiles, selected_maccs_indices)
    if global_feat is None: return None
    
    # 정규화 적용
    global_feat_scaled = scaler.transform(global_feat.reshape(1, -1)).flatten()
    
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = (torch.tensor(adj) > 0).nonzero(as_tuple=False).t().contiguous()
    global_feat_tensor = torch.tensor(global_feat_scaled, dtype=torch.float).view(-1)
    
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat_tensor)
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor)

# 전처리
print("\n최종 데이터셋 생성 중... (선택된 MACCS 키만 사용)")
data_list = [smiles_to_graph_final(s, l, top_30_maccs_indices, scaler) for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train), desc="Processing train data")]
data_list = [d for d in data_list if d is not None]

test_data = [smiles_to_graph_final(s, None, top_30_maccs_indices, scaler) for s in tqdm(smiles_test, desc="Processing test data")]
test_data = [d for d in test_data if d is not None]

# ----------------------------------------------------
#               모델 정의 (글로벌 피처 차원 수정)
# ----------------------------------------------------
# RDKit 디스크립터 20개 + 선택된 MACCS 키 30개 = 50
global_feat_dim_updated = 20 + len(top_30_maccs_indices)

class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=global_feat_dim_updated, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm1d(hidden_channels * heads)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_feat = data.global_feat.to(x.device)
        x = self.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.relu(self.bn2(self.gat2(x, edge_index)))
        x = global_mean_pool(x, batch)
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            global_feat = global_feat.view(x.size(0), -1)
        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ----------------------------------------------------
#               Optuna Objective 함수
# ----------------------------------------------------
# (이 부분은 변경 사항 없이 그대로 사용합니다)

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])

    model = GATv2WithGlobal(
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout
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
        std = np.std(trues)
        score = 0.5 * (1 - min(rmse / std, 1)) + 0.5 * corr

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
    dropout=best_params['dropout']
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn = torch.nn.SmoothL1Loss()
train_loader = DataLoader(data_list, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

# 최종 학습 루프
for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
    
    if (epoch % 20 == 0) or (epoch == 100):
        print(f"Saving model at epoch {epoch}")
        torch.save(model.state_dict(), f'./_save/gnn_rf_top30_final_epoch_{epoch}.pt')

# ----------------------------------------------------
#               최종 예측 및 저장
# ----------------------------------------------------

final_model_path = './_save/gnn_rf_top30_final_epoch_100.pt'
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