# 🔧 목적: Node + RDKit Descriptors + MACCS Keys 조합 GNN 모델

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

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')

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

def get_rdkit_and_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc).astype(np.float32)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return np.concatenate([desc, maccs_arr.astype(np.float32)])
    except:
        return None

def smiles_to_graph_with_global(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    global_feat = get_rdkit_and_maccs(smiles)
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

# 전처리
data_list = [smiles_to_graph_with_global(s, l) for s, l in zip(smiles_train, y)]
data_list = [d for d in data_list if d is not None]
test_data = [smiles_to_graph_with_global(s) for s in smiles_test]
test_data = [d for d in test_data if d is not None]

class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=187):  # 20 RDKit + 167 MACCS
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

# 학습
train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=64, shuffle=True)
val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

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
    std = np.std(trues)
    score = 0.5 * (1 - min(rmse / std, 1)) + 0.5 * corr
    print(f"[Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")
    if score > best_score:
        best_score, best_rmse, best_corr = score, rmse, corr
        counter = 0
        torch.save(model.state_dict(), './_save/gnn_node_rdkit_maccs.pt')
    else:
        counter += 1
        if counter >= patience:
            print("EarlyStopping")
            break

# 테스트 예측
model.load_state_dict(torch.load('./_save/gnn_node_rdkit_maccs2.pt'))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())
submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_node_rdkit_maccs_submission2.csv', index=False)

# 🎯 최종 결과 출력
print("\n🎯 [최종 결과 - 저장된 Best Model 기준]")
print(f"Best Score : {best_score:.4f}")
print(f"Best RMSE  : {best_rmse:.4f}")
print(f"Best Corr  : {best_corr:.4f}")
print("\n✅ 저장 완료: ./_save/gnn_node_rdkit_maccs_submission.csv")
