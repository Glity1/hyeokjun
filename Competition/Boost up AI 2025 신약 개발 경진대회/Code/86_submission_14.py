# ✅ GNN 최종 전체 코드 (수정 사항 포함)
# - BatchNorm1d → GraphNorm
# - Jumping Knowledge (mode='cat') 추가
# - Edge Feature 복수 조합 탐색 구조
# - 복합 조건 EarlyStopping 적용

import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
import torch
from torch.nn import Module, Linear, ReLU, Dropout
from torch_geometric.nn import GATv2Conv, GraphNorm, JumpingKnowledge, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import optuna

# 환경 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 데이터 로드
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')

smiles_train = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values
smiles_test = test['Canonical_Smiles'].tolist()

# SMARTS, 전기음성도, 반지름
smart_patterns = [Chem.MolFromSmarts(p) for p in ['[C]=[O]', '[NH2]', 'c1ccccc1'] if Chem.MolFromSmarts(p)]
electronegativity = {1:2.2, 6:2.55, 7:3.04, 8:3.44, 9:3.98, 15:2.19, 16:2.58, 17:3.16, 35:2.96, 53:2.66, 14:1.90}
covalent_radius = {1:0.31, 6:0.76, 7:0.71, 8:0.66, 9:0.57, 15:1.07, 16:1.05, 17:1.02, 35:1.20, 53:1.39, 14:1.11}

# RDKit Descriptor
rdkit_desc_list = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumRotatableBonds,
                   Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.HeavyAtomCount, Descriptors.FractionCSP3,
                   Descriptors.RingCount, Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
                   Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
                   Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
                   Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge]

def atom_features(atom):
    num = atom.GetAtomicNum()
    eneg = electronegativity.get(num, 0)
    radius = covalent_radius.get(num, 0)
    smarts_match = sum(atom.GetOwningMol().HasSubstructMatch(p) for p in smart_patterns)
    return [num, eneg, radius, atom.GetTotalNumHs(), int(atom.GetHybridization()), atom.GetFormalCharge(),
            int(atom.GetIsAromatic()), int(atom.IsInRing()), smarts_match]

# 기본 Edge Feature 정의

def bond_features_basic(b): return [int(b.GetBondType() == Chem.BondType.SINGLE), int(b.GetBondType() == Chem.BondType.DOUBLE),
                                     int(b.GetBondType() == Chem.BondType.TRIPLE), int(b.GetBondType() == Chem.BondType.AROMATIC),
                                     int(b.GetIsConjugated()), int(b.IsInRing()), int(b.GetStereo() == Chem.BondStereo.STEREOANY),
                                     int(b.GetStereo() == Chem.BondStereo.STEREOCIS), int(b.GetStereo() == Chem.BondStereo.STEREOTRANS)]

def bond_features_ring_info(b):
    return bond_features_basic(b) + [int(b.IsInRingSize(n)) for n in range(3, 8)]

def bond_features_all_extended(b):
    return bond_features_basic(b) + [b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum(),
                                     electronegativity.get(b.GetBeginAtom().GetAtomicNum(), 0), electronegativity.get(b.GetEndAtom().GetAtomicNum(), 0),
                                     b.GetBeginAtom().GetFormalCharge(), b.GetEndAtom().GetFormalCharge(),
                                     int(b.GetBeginAtom().GetHybridization()), int(b.GetEndAtom().GetHybridization())]

def bond_features_degree_neighbors(b):
    return bond_features_basic(b) + [b.GetBeginAtom().GetDegree(), b.GetEndAtom().GetDegree(),
                                     b.GetBeginAtom().GetTotalNumHs(), b.GetEndAtom().GetTotalNumHs()]

# 조합형 Edge Feature 생성
bond_feature_combinations = {
    'basic': [bond_features_basic],
    'ring_info': [bond_features_ring_info],
    'degree_neighbors': [bond_features_degree_neighbors],
    'all_extended': [bond_features_all_extended],
    'basic+ring_info': [bond_features_basic, bond_features_ring_info],
    'all_extended+degree_neighbors': [bond_features_all_extended, bond_features_degree_neighbors],
}

def combine_bond_features(bond, funcs):
    return sum([f(bond) for f in funcs], [])

# MACCS 기반 RandomForest Top20 선택
def get_only_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    from rdkit import DataStructs
    maccs = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs, arr)
    return arr.astype(np.float32)

maccs_data = []
y_valid = []
for s, t in zip(smiles_train, y):
    feat = get_only_maccs(s)
    if feat is not None:
        maccs_data.append(feat)
        y_valid.append(t)

maccs_data = np.array(maccs_data)
maccs_scaler = StandardScaler()
maccs_data_scaled = maccs_scaler.fit_transform(maccs_data)

rf = RandomForestRegressor(n_estimators=200, random_state=seed)
rf.fit(maccs_data_scaled, y_valid)
importances = rf.feature_importances_
top_20_maccs_indices = sorted(np.argsort(importances)[-20:])
# global feature 생성

def get_rdkit_and_selected_maccs(smiles, selected):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    desc = [f(mol) for f in rdkit_desc_list]
    from rdkit import DataStructs
    maccs = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs, arr)
    maccs_part = np.copy(arr[selected]).astype(np.float32)
    return np.nan_to_num(np.concatenate([desc, maccs_part])).astype(np.float32)

global_features_all = [get_rdkit_and_selected_maccs(s, top_20_maccs_indices) for s in smiles_train]
scaler = StandardScaler()
scaler.fit([g for g in global_features_all if g is not None])

# SMILES to Graph

def smiles_to_graph_final(smiles, label=None, selected_maccs_indices=None, scaler=None, bond_feature_extractor=None, edge_feature_dim=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    global_feat = get_rdkit_and_selected_maccs(smiles, selected_maccs_indices)
    if global_feat is None or len(global_feat) != 20 + len(selected_maccs_indices): return None
    global_feat_scaled = scaler.transform(global_feat.reshape(1, -1)).flatten()
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_feature_extractor(bond)
        edge_indices += [[u, v], [v, u]]
        edge_attrs += [feat, feat]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, edge_feature_dim), dtype=torch.float)
    global_feat_tensor = torch.tensor(global_feat_scaled, dtype=torch.float).view(-1)
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat_tensor, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, edge_attr=edge_attr)

# ✅ 모델 정의 (GraphNorm + JumpingKnowledge)
class GATv2WithGlobal_JK(Module):
    def __init__(self, node_feat_dim=9, edge_feat_dim=None, global_feat_dim=40, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.norm2 = GraphNorm(hidden_channels * heads)
        self.jk = JumpingKnowledge(mode='cat')
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(2 * hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        global_feat = data.global_feat.to(x.device)
        x1 = self.relu(self.norm1(self.gat1(x, edge_index, edge_attr)))
        x2 = self.relu(self.norm2(self.gat2(x1, edge_index, edge_attr)))
        x = self.jk([x1, x2])
        x = global_mean_pool(x, batch)
        if global_feat.dim() == 1: global_feat = global_feat.unsqueeze(0)
        if global_feat.size(0) != x.size(0): global_feat = global_feat.view(x.size(0), -1)
        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ✅ Optuna objective (복합 조건 조기 종료)
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    edge_combo_name = trial.suggest_categorical('edge_feature_combo', list(bond_feature_combinations.keys()))
    bond_feature_funcs = bond_feature_combinations[edge_combo_name]
    edge_feat_dim = len(combine_bond_features(Chem.MolFromSmiles('CC=C').GetBonds()[0], bond_feature_funcs))

    data_list = [
        smiles_to_graph_final(s, l, top_20_maccs_indices, scaler,
                              lambda b: combine_bond_features(b, bond_feature_funcs),
                              edge_feat_dim)
        for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train))
    ]
    data_list = [d for d in data_list if d is not None]

    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=batch_size)

    model = GATv2WithGlobal_JK(hidden_channels=hidden_channels, heads=heads, dropout=dropout,
                                edge_feat_dim=edge_feat_dim, global_feat_dim=20 + len(top_20_maccs_indices)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_score = -np.inf
    best_rmse = np.inf
    best_corr = -np.inf
    patience_counter = 0

    for epoch in range(1, 101):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y.view(-1))
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
        norm_rmse = rmse / y_range if y_range > 0 else 1.0
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

        trial.report(score, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        if (score > best_score) or (rmse < best_rmse and corr > best_corr):
            best_score, best_rmse, best_corr = score, rmse, corr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_score

# ✅ Optuna 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)
print("\n\nBest Parameters:", study.best_params)
print("Best Score:", study.best_value)

# ✅ 최적 파라미터 기반 최종 학습 및 예측
best_params = study.best_params
combo_name = best_params['edge_feature_combo']
bond_funcs = bond_feature_combinations[combo_name]
edge_feat_dim = len(combine_bond_features(Chem.MolFromSmiles('CC=C').GetBonds()[0], bond_funcs))

final_data_list = [
    smiles_to_graph_final(s, l, top_20_maccs_indices, scaler,
                          lambda b: combine_bond_features(b, bond_funcs), edge_feat_dim)
    for s, l in tqdm(zip(smiles_train, y), desc="Final Train Graph")
]
final_data_list = [d for d in final_data_list if d is not None]

final_test_data = [
    smiles_to_graph_final(s, None, top_20_maccs_indices, scaler,
                          lambda b: combine_bond_features(b, bond_funcs), edge_feat_dim)
    for s in tqdm(smiles_test, desc="Final Test Graph")
]
final_test_data = [d for d in final_test_data if d is not None]

train_idx, val_idx = train_test_split(range(len(final_data_list)), test_size=0.2, random_state=seed)
train_loader = DataLoader([final_data_list[i] for i in train_idx], batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader([final_data_list[i] for i in val_idx], batch_size=best_params['batch_size'])
test_loader = DataLoader(final_test_data, batch_size=best_params['batch_size'])

model = GATv2WithGlobal_JK(
    hidden_channels=best_params['hidden_channels'],
    heads=best_params['heads'],
    dropout=best_params['dropout'],
    edge_feat_dim=edge_feat_dim,
    global_feat_dim=20 + len(top_20_maccs_indices)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn = torch.nn.MSELoss()

best_score = -np.inf
best_rmse = np.inf
best_corr = -np.inf
patience_counter = 0

for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch.y.view(-1))
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
    norm_rmse = rmse / y_range if y_range > 0 else 1.0
    score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

    print(f"Epoch {epoch:03d} | Score: {score:.4f} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")

    if (score > best_score) or (rmse < best_rmse and corr > best_corr):
        best_score, best_rmse, best_corr = score, rmse, corr
        patience_counter = 0
        torch.save(model.state_dict(), './_save/gnn_20optimized_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 20:
            print("Early Stopping")
            break

# ✅ Test 예측 및 저장
model.load_state_dict(torch.load('./_save/gnn_20optimized_model.pt'))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_preds.append(pred.cpu().numpy())

submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_20optimized_submission.csv', index=False)
print("\n✅ 최종 제출 파일 저장 완료: ./_save/gnn_20optimized_submission.csv")