import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import torch
from torch.nn import Module, Linear, ReLU, Dropout, BatchNorm1d
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GINEConv, global_mean_pool
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- 설정 ---
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 로딩 ---
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')

smiles_train = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values
smiles_test = test['Canonical_Smiles'].tolist()

# --- RDKit Descriptor 정의 ---
rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, Descriptors.RingCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge
]
rdkit_desc_names = [f.__name__ for f in rdkit_desc_list]

# --- RDKit Descriptor 추출 함수 ---
def extract_rdkit_descriptors(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([np.nan] * len(rdkit_desc_list))
            continue
        desc = [f(mol) for f in rdkit_desc_list]
        features.append(desc)
    return pd.DataFrame(features, columns=rdkit_desc_names)

# --- VIF 계산 및 다중공선성 제거 ---
def calculate_vif(df):
    df = df.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(scaled, i) for i in range(scaled.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)

rdkit_df = extract_rdkit_descriptors(smiles_train)
vif_result = calculate_vif(rdkit_df)
selected_rdkit_cols = vif_result[vif_result['VIF'] < 5]['feature'].tolist()
print(f"\n✅ VIF < 5 기준 선택된 RDKit 특성: {len(selected_rdkit_cols)}개")

# --- Global Feature 생성 함수 ---
def get_global_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        # RDKit
        desc_all = [f(mol) for f in rdkit_desc_list]
        desc_dict = dict(zip(rdkit_desc_names, desc_all))
        selected = [desc_dict[col] for col in selected_rdkit_cols]
        selected = np.nan_to_num(selected).astype(np.float32)

        # MACCS
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

        return np.concatenate([selected, maccs_arr.astype(np.float32)])
    except:
        return None

# --- Graph 변환 함수 (GATv2 / GINE 공용) ---
def atom_features(atom):
    electronegativity = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58,
                         17: 3.16, 35: 2.96, 53: 2.66, 14: 1.90}
    covalent_radius = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05,
                       17: 1.02, 35: 1.20, 53: 1.39, 14: 1.11}
    num = atom.GetAtomicNum()
    eneg = electronegativity.get(num, 0)
    radius = covalent_radius.get(num, 0)
    return [
        num, eneg, radius,
        atom.GetTotalNumHs(), int(atom.GetHybridization()), atom.GetFormalCharge(),
        int(atom.GetIsAromatic()), int(atom.IsInRing()), int(atom.GetDegree())
    ]

def bond_features(bond):
    bond_type = bond.GetBondType()
    feats = [
        int(bond_type == Chem.BondType.SINGLE),
        int(bond_type == Chem.BondType.DOUBLE),
        int(bond_type == Chem.BondType.TRIPLE),
        int(bond_type == Chem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    return feats

def smiles_to_graph_with_global(smiles, label=None, include_edge_attr=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    global_feat = get_global_features(smiles)
    if global_feat is None: return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        edge_indices.append([u, v])
        edge_indices.append([v, u])
        edge_features.append(feats)
        edge_features.append(feats)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if include_edge_attr else None
    global_feat = torch.tensor(global_feat, dtype=torch.float).view(-1)
    y_tensor = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat, y=y_tensor)


# --- 3. GNN 모델 정의 ---

# 3.1 GATv2WithGlobal 모델 (기존 모델)
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=len(selected_rdkit_cols) + 167):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, 64, heads=4, dropout=0.2)
        self.bn1 = BatchNorm1d(64 * 4)
        self.gat2 = GATv2Conv(64 * 4, 128, heads=4, dropout=0.2)
        self.bn2 = BatchNorm1d(128 * 4)
        self.relu = ReLU()
        self.dropout = Dropout(0.3)
        self.fc1 = Linear(128 * 4 + global_feat_dim, 128)
        self.fc2 = Linear(128, 1)

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

# 3.2 GINEWithGlobal 모델 (엣지 피처 활용) - 오류 수정 반영
class GINEWithGlobal(Module):
    def __init__(self, node_feat_dim=9, edge_feat_dim=6, global_feat_dim= len(selected_rdkit_cols) + 167):
        super().__init__()
        
        # GINEConv의 메시지 함수로 사용될 MLP 정의
        # GINEConv는 nn에 입력으로 노드 피처만을 전달합니다.
        # edge_attr은 별도로 GINEConv의 edge_dim 매개변수로 처리됩니다.
        self.nn1 = nn.Sequential(
            nn.Linear(node_feat_dim, 64), # 노드 피처 차원(9)이 입력이 됩니다.
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.gine1 = GINEConv(self.nn1, edge_dim=edge_feat_dim) # <-- edge_dim 설정 추가
        self.bn1 = BatchNorm1d(64)

        self.nn2 = nn.Sequential(
            nn.Linear(64, 128),            # 이전 레이어 출력(64)이 입력이 됩니다.
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.gine2 = GINEConv(self.nn2, edge_dim=edge_feat_dim) # <-- edge_dim 설정 추가
        self.bn2 = BatchNorm1d(128)
        
        self.relu = ReLU()
        self.dropout = Dropout(0.3)
        
        self.fc1 = Linear(128 + global_feat_dim, 128) 
        self.fc2 = Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        global_feat = data.global_feat.to(x.device)

        # GINEConv 호출 시 edge_attr을 매개변수로 전달
        x = self.relu(self.bn1(self.gine1(x, edge_index, edge_attr)))
        x = self.relu(self.bn2(self.gine2(x, edge_index, edge_attr)))

        x = global_mean_pool(x, batch)
        
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            global_feat = global_feat.view(x.size(0), -1)
            
        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# --- 4. 학습 및 평가 함수 ---
def train_and_evaluate_model(model, train_loader, val_loader, model_name, epochs=100, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.SmoothL1Loss()
    
    best_score, best_rmse, best_corr = -np.inf, 0, 0
    counter = 0

    print(f"\n--- Training {model_name} ---")
    for epoch in range(1, epochs + 1):
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
        normalized_rmse = rmse / y_range
        score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)
        
        print(f"[{model_name} Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")
        
        if score > best_score:
            best_score, best_rmse, best_corr = score, rmse, corr
            counter = 0
            torch.save(model.state_dict(), f'./_save/{model_name}.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"EarlyStopping for {model_name}")
                break
    print(f"--- {model_name} Training Finished ---")
    print(f"Best Score: {best_score:.4f}, Best RMSE: {best_rmse:.4f}, Best Corr: {best_corr:.4f}")
    
    return model

def predict_with_model(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_preds.append(out.cpu().numpy())
    return np.concatenate(test_preds)

# --- 5. 데이터 전처리 (GATv2와 GINE 각각에 맞게) ---
# GATv2는 edge_attr을 사용하지 않으므로 include_edge_attr=False
data_list_gatv2 = [smiles_to_graph_with_global(s, l, include_edge_attr=False) for s, l in tqdm(zip(smiles_train, y), desc="Processing for GATv2")]
data_list_gatv2 = [d for d in data_list_gatv2 if d is not None]

test_data_gatv2 = [smiles_to_graph_with_global(s, include_edge_attr=False) for s in tqdm(smiles_test, desc="Processing Test for GATv2")]
test_data_gatv2 = [d for d in test_data_gatv2 if d is not None]

# GINE는 edge_attr을 사용하므로 include_edge_attr=True
data_list_gine = [smiles_to_graph_with_global(s, l, include_edge_attr=True) for s, l in tqdm(zip(smiles_train, y), desc="Processing for GINE")]
data_list_gine = [d for d in data_list_gine if d is not None]

test_data_gine = [smiles_to_graph_with_global(s, include_edge_attr=True) for s in tqdm(smiles_test, desc="Processing Test for GINE")]
test_data_gine = [d for d in test_data_gine if d is not None]


# --- 6. 학습 및 예측 실행 ---

# GATv2 모델 학습 및 저장
train_idx_gatv2, val_idx_gatv2 = train_test_split(range(len(data_list_gatv2)), test_size=0.2, random_state=seed)
train_loader_gatv2 = DataLoader([data_list_gatv2[i] for i in train_idx_gatv2], batch_size=64, shuffle=True)
val_loader_gatv2 = DataLoader([data_list_gatv2[i] for i in val_idx_gatv2], batch_size=64)
test_loader_gatv2 = DataLoader(test_data_gatv2, batch_size=64)

model_gatv2 = GATv2WithGlobal().to(device)
print(f"GATv2 model on device: {device}")
model_gatv2 = train_and_evaluate_model(model_gatv2, train_loader_gatv2, val_loader_gatv2, 'gnn_gatv2_rdkit_maccs')
gatv2_preds = predict_with_model(model_gatv2, test_loader_gatv2, './_save/gnn_gatv2_rdkit_maccs.pt')


# GINE 모델 학습 및 저장
train_idx_gine, val_idx_gine = train_test_split(range(len(data_list_gine)), test_size=0.2, random_state=seed)
train_loader_gine = DataLoader([data_list_gine[i] for i in train_idx_gine], batch_size=64, shuffle=True)
val_loader_gine = DataLoader([data_list_gine[i] for i in val_idx_gine], batch_size=64)
test_loader_gine = DataLoader(test_data_gine, batch_size=64)

model_gine = GINEWithGlobal().to(device)
print(f"GINE model on device: {device}")
model_gine = train_and_evaluate_model(model_gine, train_loader_gine, val_loader_gine, 'gnn_gine_rdkit_maccs')
gine_preds = predict_with_model(model_gine, test_loader_gine, './_save/gnn_gine_rdkit_maccs.pt')


# --- 7. 앙상블 및 최종 제출 ---
print("\n--- Performing Ensemble Prediction ---")

if len(gatv2_preds) != len(gine_preds):
    print("경고: GATv2와 GINE의 테스트 예측 길이가 다릅니다. 이로 인해 앙상블 결과가 부정확할 수 있습니다.")
    print("정확한 앙상블을 위해서는 Data 객체 생성 시 원본 인덱스를 포함하고, 누락된 예측값을 처리해야 합니다.")
    # 임시 방편으로 더 짧은 길이에 맞춤
    min_len = min(len(gatv2_preds), len(gine_preds))
    ensemble_preds = (gatv2_preds[:min_len] + gine_preds[:min_len]) / 2
else:
    ensemble_preds = (gatv2_preds + gine_preds) / 2


# Submit DataFrame에 결과 채우기
submit_df['Inhibition'] = ensemble_preds
submit_df.to_csv('./_save/gnn_ensemble_submission.csv', index=False)

print("\n🎯 [앙상블 최종 결과]")
print(f"GATv2 예측 결과 수: {len(gatv2_preds)}")
print(f"GINE 예측 결과 수: {len(gine_preds)}")
print(f"앙상블된 예측 결과 수: {len(ensemble_preds)}")
print("\n✅ 앙상블 결과 저장 완료: ./_save/gnn_ensemble_submission.csv")

import optuna
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 앙상블 Score 계산 함수
def calc_ensemble_score(preds, trues):
    rmse = np.sqrt(mean_squared_error(trues, preds))
    corr = pearsonr(trues, preds)[0]
    y_range = trues.max() - trues.min()
    normalized_rmse = rmse / y_range
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)
    return score, rmse, corr

# 앙상블 튜닝용 validation 데이터셋 생성
val_preds_gatv2 = []
val_trues = []
model_gatv2.load_state_dict(torch.load('./_save/gnn_gatv2_rdkit_maccs.pt'))
model_gatv2.eval()
with torch.no_grad():
    for batch in val_loader_gatv2:
        batch = batch.to(device)
        out = model_gatv2(batch)
        val_preds_gatv2.append(out.cpu().numpy())
        val_trues.append(batch.y.view(-1).cpu().numpy())
val_preds_gatv2 = np.concatenate(val_preds_gatv2)
val_trues = np.concatenate(val_trues)

val_preds_gine = []
model_gine.load_state_dict(torch.load('./_save/gnn_gine_rdkit_maccs.pt'))
model_gine.eval()
with torch.no_grad():
    for batch in val_loader_gine:
        batch = batch.to(device)
        out = model_gine(batch)
        val_preds_gine.append(out.cpu().numpy())
val_preds_gine = np.concatenate(val_preds_gine)

# Optuna Objective
def objective(trial):
    w = trial.suggest_float("weight", 0.0, 1.0)
    ensemble = w * val_preds_gatv2 + (1 - w) * val_preds_gine
    score, rmse, corr = calc_ensemble_score(ensemble, val_trues)
    return -score  # maximize score → minimize -score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_weight = study.best_params["weight"]
print(f"\n✅ 최적 가중치 (GATv2): {best_weight:.4f}, (GINE): {1 - best_weight:.4f}")

# 최적 가중치로 Test 예측 앙상블
final_preds = best_weight * gatv2_preds + (1 - best_weight) * gine_preds
submit_df['Inhibition'] = final_preds
submit_df.to_csv('./_save/gnn_ensemble_weighted_submission.csv', index=False)

print("\n✅ 최종 가중치 앙상블 제출 완료: ./_save/gnn_ensemble_weighted_submission.csv")

# EarlyStopping for gnn_gine_rdkit_maccs
# --- gnn_gine_rdkit_maccs Training Finished ---
# Best Score: 0.6172, Best RMSE: 24.6256, Best Corr: 0.4822