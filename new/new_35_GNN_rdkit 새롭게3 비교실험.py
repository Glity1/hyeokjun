import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import os

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values
smiles_train = train['Canonical_Smiles'].tolist()
smiles_test = test['Canonical_Smiles'].tolist()

# ✅ 기존 RDKit top20 특성 정의
rdkit_top20_names = [
    'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors',
    'HeavyAtomCount', 'FractionCSP3', 'RingCount', 'NHOHCount', 'NOCount',
    'NumAliphaticRings', 'NumAromaticRings', 'NumSaturatedRings', 'ExactMolWt',
    'MolMR', 'LabuteASA', 'BalabanJ', 'MaxPartialCharge', 'MinPartialCharge'
]

# ✅ 중요도 기반 추가 RDKit 특성 로드
importance_df = pd.read_csv('./_data/dacon/new/rdkit_additional_top_features.csv')
importance_df = importance_df[~importance_df['Feature'].isin(rdkit_top20_names)].reset_index(drop=True)

# ✅ 전체 RDKit descriptor dict
rdkit_all = {desc[0]: desc[1] for desc in Descriptors.descList}

# ✅ GNN + Global 모델 정의
class GATv2WithGlobal(nn.Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=20):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, 64, heads=4, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(64 * 4)
        self.gat2 = GATv2Conv(64 * 4, 128, heads=4, dropout=0.2)
        self.bn2 = nn.BatchNorm1d(128 * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 + global_feat_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_feat = data.global_feat.to(x.device)

        x = self.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.relu(self.bn2(self.gat2(x, edge_index)))
        x = global_mean_pool(x, batch)

        # 🔧 [여기 추가]
        if global_feat.dim() == 1 or global_feat.size(0) != x.size(0):
            global_feat = global_feat.view(x.size(0), -1)

        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ✅ global feature 추출 함수
def get_global_features(smiles, descriptor_names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    values = []
    for name in descriptor_names:
        try:
            val = rdkit_all[name](mol)
        except:
            val = 0.0
        values.append(val)
    return np.nan_to_num(values, nan=0.0)

# ✅ graph + global 구성 함수
def smiles_to_graph(smiles, label=None, descriptor_names=[]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Node feature
    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(), atom.GetTotalNumHs(), int(atom.GetHybridization()),
            atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
            atom.GetDegree(), atom.GetMass(), atom.GetImplicitValence()
        ])
    x = torch.tensor(x, dtype=torch.float)
    # Edge
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    row, col = np.array(adj).nonzero()
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    # Global
    global_feat = get_global_features(smiles, descriptor_names)
    if global_feat is None:
        return None
    global_feat = torch.tensor(global_feat, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, global_feat=global_feat)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data

# ✅ 루프 기반 실험
results = []
for top_n in [10, 30, 50]:
    print(f"\n🔍 실험 시작: 추가 Top{top_n} 적용")
    descriptor_names = rdkit_top20_names + importance_df['Feature'][:top_n].tolist()
    data_list = [smiles_to_graph(smi, lbl, descriptor_names) for smi, lbl in zip(smiles_train, y)]
    data_list = [d for d in data_list if d is not None]

    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=64, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=64)

    model = GATv2WithGlobal(global_feat_dim=len(descriptor_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()

    best_score = -np.inf
    counter = 0
    patience = 20
    for epoch in range(1, 301):
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
        norm_rmse = rmse / (trues.max() - trues.min())
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)
        if score > best_score:
            best_score = score
            best_rmse = rmse
            best_corr = corr
            torch.save(model.state_dict(), f'./_save/gnn_rdkit_top{top_n}.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"✅ EarlyStopping @Epoch {epoch}")
                break
        if epoch % 20 == 0:
            print(f"[Top{top_n}][Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")

    results.append((top_n, best_score, best_rmse, best_corr))

# ✅ 결과 출력
print("\n🎯 [최종 비교 결과]")
for top_n, score, rmse, corr in results:
    print(f"Top{top_n:2d} ➤ Score: {score:.4f} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")
