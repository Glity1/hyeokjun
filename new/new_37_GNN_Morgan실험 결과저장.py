import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
from torch.nn import Module, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./_save", exist_ok=True)

# ✅ 데이터 로드
train_df = pd.read_csv("./_data/dacon/new/train.csv").drop_duplicates("Canonical_Smiles").reset_index(drop=True)
test_df = pd.read_csv("./_data/dacon/new/test.csv")
submit_df = pd.read_csv("./_data/dacon/new/sample_submission.csv")

smiles_train = train_df["Canonical_Smiles"].tolist()
smiles_test = test_df["Canonical_Smiles"].tolist()
y = train_df["Inhibition"].values

# ✅ Morgan Fingerprint (2048bit)
def get_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

morgan_train = np.array([get_morgan_fp(s) for s in smiles_train])
morgan_test = np.array([get_morgan_fp(s) for s in smiles_test])
valid_idx = [i for i in range(len(morgan_train)) if morgan_train[i] is not None]
morgan_train = morgan_train[valid_idx]
y = y[valid_idx]

# ✅ Top150 인덱스 저장 (최초 1회만)
rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
rf.fit(morgan_train, y)
importances = rf.feature_importances_
top150_indices = np.argsort(importances)[::-1][:150]  # 중요도 기준 내림차순
np.save("./_save/top150_morgan_indices.npy", top150_indices)
print("✅ Top150 인덱스 저장 완료: ./_save/top150_morgan_indices.npy")

# ✅ 이후 실험에서 불러와 상위 N개 선택
top150_indices = np.load("./_save/top150_morgan_indices.npy")
exit()
# ✅ 실험할 Top-N 목록
for N in [30, 50]:
    print(f"\n🚀 Morgan Top-{N} 실험 시작")
    selected_indices = top150_indices[:N]
    selected_train = morgan_train[:, selected_indices]
    selected_test = morgan_test[:, selected_indices]

    # ✅ 노드 피처
    def atom_features(atom):
        return [
            atom.GetAtomicNum(), atom.GetTotalNumHs(), int(atom.GetHybridization()),
            atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
            atom.GetDegree(), atom.GetExplicitValence(), atom.GetImplicitValence()
        ]

    # ✅ 그래프 생성
    def smiles_to_graph(smiles, global_feat, label=None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or global_feat is None: return None
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
        adj = Chem.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(adj).nonzero(), dtype=torch.long)
        global_feat = torch.tensor(global_feat, dtype=torch.float).view(-1)
        if label is not None:
            y = torch.tensor([label], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat)
        else:
            return Data(x=x, edge_index=edge_index, global_feat=global_feat)

    smiles_valid = [smiles_train[i] for i in valid_idx]
    data_list = [smiles_to_graph(s, f, l) for s, f, l in zip(smiles_valid, selected_train, y)]
    data_list = [d for d in data_list if d is not None]
    test_data = [smiles_to_graph(s, f) for s, f in zip(smiles_test, selected_test)]
    test_data = [d for d in test_data if d is not None]

    # ✅ 모델 정의
    class GATv2WithGlobal(Module):
        def __init__(self, node_feat_dim=9, global_feat_dim=N):
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

    # ✅ 학습 루프
    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=64, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=64)

    model = GATv2WithGlobal().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.SmoothL1Loss()
    best_score = -np.inf
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
        norm_rmse = rmse / (trues.max() - trues.min())
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

        print(f"[Top-{N} Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")

        if score > best_score:
            best_score = score
            counter = 0
            torch.save(model.state_dict(), f"./_save/gnn_morgan_top{N}.pt")
        else:
            counter += 1
            if counter >= patience:
                print("EarlyStopping")
                break

    print(f"✅ Top-{N} 최종 성능 | Score: {score:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")
