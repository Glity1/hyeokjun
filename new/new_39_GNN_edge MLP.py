# ✅ 목적: Bond Stereo를 One-hot 인코딩하여 edge feature로 포함한 GNN 전체 코드

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

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')

smiles_train = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values
smiles_test = test['Canonical_Smiles'].tolist()

# ✅ Morgan FP
def get_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

morgan_train = np.array([get_morgan_fp(s) for s in smiles_train])
morgan_test = np.array([get_morgan_fp(s) for s in smiles_test])

valid_idx = [i for i in range(len(morgan_train)) if morgan_train[i] is not None]
smiles_train = [smiles_train[i] for i in valid_idx]
y = y[valid_idx]
morgan_train = morgan_train[valid_idx]

# ✅ Morgan TopN
use_topN = 30
top_idx_path = './_save/morgan_top30_from_combined.npy' if use_topN == 30 else './_save/morgan_top50_from_combined.npy'
top_idx = np.load(top_idx_path)
morgan_train_top = morgan_train[:, top_idx]
morgan_test_top = morgan_test[:, top_idx]

# ✅ RDKit + MACCS
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
    if mol is None: return None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc).astype(np.float32)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return np.concatenate([desc, maccs_arr.astype(np.float32)])
    except:
        return None

# ✅ Node Feature
def atom_features(atom):
    return [
        atom.GetAtomicNum(), atom.GetTotalNumHs(), int(atom.GetHybridization()),
        atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
        atom.GetDegree(), atom.GetExplicitValence(), atom.GetImplicitValence(),
    ]

# ✅ Stereo One-hot
def stereo_onehot(bond):
    onehot = [0] * 5
    stereo = int(bond.GetStereo())
    if 0 <= stereo < 5:
        onehot[stereo] = 1
    return onehot

# ✅ Edge Feature (9개)
def get_bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.GetIsAromatic(),
        *stereo_onehot(bond),
        bond.IsInRing(),
    ]

# ✅ Graph 변환
def smiles_to_graph_with_global(smiles, global_feat, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    edge_index = torch.tensor(edge_indices).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    global_feat = torch.tensor(global_feat, dtype=torch.float).view(-1)
    y = torch.tensor([label], dtype=torch.float) if label is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat, y=y)

# ✅ 통합 피처 구성
X_train, X_test = [], []
for i, s in enumerate(smiles_train):
    rdkit_maccs = get_rdkit_and_maccs(s)
    if rdkit_maccs is None: continue
    global_feat = np.concatenate([rdkit_maccs, morgan_train_top[i]])
    X_train.append((s, global_feat, y[i]))

for i, s in enumerate(smiles_test):
    rdkit_maccs = get_rdkit_and_maccs(s)
    if rdkit_maccs is None: continue
    global_feat = np.concatenate([rdkit_maccs, morgan_test_top[i]])
    X_test.append((s, global_feat))

train_data = [smiles_to_graph_with_global(s, f, l) for s, f, l in X_train]
test_data = [smiles_to_graph_with_global(s, f) for s, f in X_test]
train_data = [d for d in train_data if d is not None]
test_data = [d for d in test_data if d is not None]

# ✅ GNN 모델 정의 (edge_dim=9)
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=187 + use_topN):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, 64, heads=4, dropout=0.2, edge_dim=9)
        self.bn1 = BatchNorm1d(64 * 4)
        self.gat2 = GATv2Conv(64 * 4, 128, heads=4, dropout=0.2, edge_dim=9)
        self.bn2 = BatchNorm1d(128 * 4)
        self.relu = ReLU()
        self.dropout = Dropout(0.3)
        self.fc1 = Linear(128 * 4 + global_feat_dim, 128)
        self.fc2 = Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_feat = data.global_feat.to(x.device)
        edge_attr = edge_attr.to(x.device)
        x = self.relu(self.bn1(self.gat1(x, edge_index, edge_attr)))
        x = self.relu(self.bn2(self.gat2(x, edge_index, edge_attr)))
        x = global_mean_pool(x, batch)
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            global_feat = global_feat.view(x.size(0), -1)
        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ✅ 학습 준비
train_idx, val_idx = train_test_split(range(len(train_data)), test_size=0.2, random_state=seed)
train_loader = DataLoader([train_data[i] for i in train_idx], batch_size=64, shuffle=True)
val_loader = DataLoader([train_data[i] for i in val_idx], batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# ✅ 학습
model = GATv2WithGlobal().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.SmoothL1Loss()
best_score, best_rmse, best_corr = -np.inf, 0, 0
counter, patience = 0, 20

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
    normalized_rmse = rmse / y_range
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)
    print(f"[Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")
    if score > best_score:
        best_score, best_rmse, best_corr = score, rmse, corr
        counter = 0
        torch.save(model.state_dict(), f'./_save/gnn_edge9_stereo1hot_morgan{use_topN}.pt')
    else:
        counter += 1
        if counter >= patience:
            print("EarlyStopping")
            break

# ✅ 테스트 예측
model.load_state_dict(torch.load(f'./_save/gnn_edge9_stereo1hot_morgan{use_topN}.pt'))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())
submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv(f'./_save/gnn_edge9_stereo1hot_morgan{use_topN}_submission.csv', index=False)

# 🎯 결과 출력
print("\n🎯 [최종 결과 - 저장된 Best Model 기준]")
print(f"Best Score : {best_score:.4f}")
print(f"Best RMSE  : {best_rmse:.4f}")
print(f"Best Corr  : {best_corr:.4f}")
print("\n✅ 저장 완료:", f'./_save/gnn_edge9_stereo1hot_morgan{use_topN}_submission.csv')
