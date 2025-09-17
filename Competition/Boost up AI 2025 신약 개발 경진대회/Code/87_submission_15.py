import numpy as np, pandas as pd, os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import shap, matplotlib.pyplot as plt
import torch
from torch.nn import Module, Linear, ReLU, Dropout
from torch_geometric.nn import GATv2Conv, GraphNorm, JumpingKnowledge, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import optuna

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 데이터
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
smiles_train = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values
smiles_test = test['Canonical_Smiles'].tolist()

# RDKit Descriptor
rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumRotatableBonds,
    Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.HeavyAtomCount, Descriptors.FractionCSP3,
    Descriptors.RingCount, Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge,
]
rdkit_desc_names = [f.__name__ for f in rdkit_desc_list]

def calc_rdkit_features(smiles_list):
    data = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        data.append([f(mol) if mol else np.nan for f in rdkit_desc_list])
    return pd.DataFrame(data, columns=rdkit_desc_names)

def remove_multicollinearity(df, threshold=0.95):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop), to_drop

# SHAP 기반 RDKit 중요도 정제
rdkit_df = calc_rdkit_features(smiles_train)
rdkit_df_clean = rdkit_df.dropna()
y_clean = np.array(y)[rdkit_df_clean.index]
rdkit_df_nocol, dropped = remove_multicollinearity(rdkit_df_clean)

scaler_rdkit = StandardScaler()
X_scaled = scaler_rdkit.fit_transform(rdkit_df_nocol)
xgb = XGBRegressor(n_estimators=200, random_state=seed)
xgb.fit(X_scaled, y_clean)
explainer = shap.Explainer(xgb)
shap_values = explainer(X_scaled)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_n_rdkit = 15
top_rdkit_cols = rdkit_df_nocol.columns[np.argsort(mean_abs_shap)[-top_n_rdkit:]].tolist()

# 시각화
top_n_rdkit_actual = len(top_rdkit_cols)
plt.figure(figsize=(10, 6))
plt.barh(range(top_n_rdkit_actual), sorted(mean_abs_shap[np.argsort(mean_abs_shap)[-top_n_rdkit_actual:]]))
plt.yticks(range(top_n_rdkit_actual), top_rdkit_cols[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP 중요도 기반 RDKit 상위 특성")
plt.tight_layout()
plt.show()

# MACCS Top20
def get_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    from rdkit import DataStructs
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

maccs_arr, y_valid = [], []
for s, t in zip(smiles_train, y):
    m = get_maccs(s)
    if m is not None: maccs_arr.append(m); y_valid.append(t)
maccs_arr = np.array(maccs_arr)
rf = RandomForestRegressor(n_estimators=200, random_state=seed)
rf.fit(maccs_arr, y_valid)
top20_maccs_idx = sorted(np.argsort(rf.feature_importances_)[-20:])

# SMARTS + 전기음성도 + 반지름
smart_patterns = [Chem.MolFromSmarts(p) for p in ['[C]=[O]', '[NH2]', 'c1ccccc1'] if Chem.MolFromSmarts(p)]
electronegativity = {1:2.2, 6:2.55, 7:3.04, 8:3.44, 9:3.98, 15:2.19, 16:2.58, 17:3.16, 35:2.96, 53:2.66, 14:1.90}
covalent_radius = {1:0.31, 6:0.76, 7:0.71, 8:0.66, 9:0.57, 15:1.07, 16:1.05, 17:1.02, 35:1.20, 53:1.39, 14:1.11}

def atom_features(atom):
    num = atom.GetAtomicNum()
    return [num,
            electronegativity.get(num, 0),
            covalent_radius.get(num, 0),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            sum(atom.GetOwningMol().HasSubstructMatch(p) for p in smart_patterns)]

def get_global_feature(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    desc_dict = dict(zip(rdkit_desc_names, [f(mol) for f in rdkit_desc_list]))
    rdkit_vals = [desc_dict[k] for k in top_rdkit_cols]
    from rdkit import DataStructs
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    maccs_vals = np.copy(arr[top20_maccs_idx]).astype(np.float32)
    return np.nan_to_num(np.concatenate([rdkit_vals, maccs_vals])).astype(np.float32)

# Edge Feature 정의
def bond_features_basic(b): return [int(b.GetBondType() == Chem.BondType.SINGLE), int(b.GetBondType() == Chem.BondType.DOUBLE),
                                     int(b.GetBondType() == Chem.BondType.TRIPLE), int(b.GetBondType() == Chem.BondType.AROMATIC),
                                     int(b.GetIsConjugated()), int(b.IsInRing())]

# Graph 생성
def smiles_to_graph(smiles, label=None, scaler=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    global_feat = get_global_feature(smiles)
    if global_feat is None or len(global_feat) != len(top_rdkit_cols) + len(top20_maccs_idx): return None
    global_feat_scaled = scaler.transform(global_feat.reshape(1, -1)).flatten()
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = bond_features_basic(b)
        edge_indices += [[u, v], [v, u]]
        edge_attrs += [feat, feat]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    global_feat_tensor = torch.tensor(global_feat_scaled, dtype=torch.float)
    if label is not None:
        return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float),
                    edge_attr=edge_attr, global_feat=global_feat_tensor)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat_tensor)

# Global Scaler
global_train_feat = [get_global_feature(s) for s in smiles_train if get_global_feature(s) is not None]
scaler = StandardScaler().fit(global_train_feat)

# GNN 모델
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, edge_feat_dim=6, global_feat_dim=35, hidden_channels=128, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, edge_dim=edge_feat_dim)
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_feat_dim)
        self.norm2 = GraphNorm(hidden_channels * heads)
        self.jk = JumpingKnowledge(mode='cat')
        self.fc1 = Linear(2 * hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
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

# 학습 데이터 생성
graph_data = [smiles_to_graph(s, l, scaler) for s, l in zip(smiles_train, y)]
graph_data = [d for d in graph_data if d is not None]
train_idx, val_idx = train_test_split(range(len(graph_data)), test_size=0.2, random_state=seed)
train_loader = DataLoader([graph_data[i] for i in train_idx], batch_size=64, shuffle=True)
val_loader = DataLoader([graph_data[i] for i in val_idx], batch_size=64)

# 모델 초기화
model = GATv2WithGlobal(global_feat_dim=len(top_rdkit_cols)+len(top20_maccs_idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
loss_fn = torch.nn.MSELoss()

# 학습
best_score = -np.inf
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
    score = 0.5 * (1 - min(rmse / (trues.max() - trues.min()), 1)) + 0.5 * np.clip(corr, 0, 1)
    print(f"[{epoch:03d}] Score: {score:.4f} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), './_save/gnn_rdkitshap.pt')

# Test 예측
test_graphs = [smiles_to_graph(s, None, scaler) for s in smiles_test]
test_graphs = [d for d in test_graphs if d is not None]
test_loader = DataLoader(test_graphs, batch_size=64)
model.load_state_dict(torch.load('./_save/gnn_rdkitshap.pt'))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_preds.append(pred.cpu().numpy())
submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_rdkitshap_submission.csv', index=False)
print("✅ 제출 저장 완료: ./_save/gnn_rdkitshap_submission.csv")
