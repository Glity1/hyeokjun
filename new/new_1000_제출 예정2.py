import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm, JumpingKnowledge
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
import optuna
import joblib
import shap
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module='rdkit.Chem.MACCSkeys')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./_save", exist_ok=True)

# 1. 📦 데이터 로드
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submission = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# 2. 🧪 RDKit Descriptors + MACCS Keys 계산 함수
def calc_rdkit_maccs(smiles_list):
    rdkit_desc_list = []
    maccs_keys_list = []
    desc_names = [d[0] for d in Descriptors._descList]
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rdkit_desc = [desc(mol) for _, desc in Descriptors._descList]
            maccs = list(MACCSkeys.GenMACCSKeys(mol))
        else:
            rdkit_desc = [0] * len(desc_names)
            maccs = [0] * 167
        rdkit_desc_list.append(rdkit_desc)
        maccs_keys_list.append(maccs)
    rdkit_arr = np.array(rdkit_desc_list)
    maccs_arr = np.array(maccs_keys_list)
    return rdkit_arr, maccs_arr

# 3. 🧬 GNN 입력 변환
def mol_to_graph_data_obj(smiles, global_feat):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs()
        ])
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_feats, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.global_feat = torch.tensor(global_feat, dtype=torch.float)
    return data

# 4. 📊 평가 함수
def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    score = 0.5 * (1 - min(rmse / (max(y_true) - min(y_true)), 1)) + 0.5 * max(corr, 0)
    return rmse, corr, score

# 5. 🧠 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, in_dim, global_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden))
        self.norms.append(GraphNorm(hidden))
        for _ in range(layers - 1):
            self.convs.append(GATv2Conv(hidden, hidden))
            self.norms.append(GraphNorm(hidden))
        self.jump = JumpingKnowledge("cat")
        self.fc1 = Linear(hidden * layers + global_dim, hidden)
        self.fc2 = Linear(hidden, 1)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        outs = []
        for conv, norm in zip(self.convs, self.norms):
            x = self.relu(norm(conv(x, edge_index)))
            outs.append(global_mean_pool(x, batch))
        
        h = torch.cat(outs, dim=1)
        g = data.global_feat.to(h.device)

        # ======================================================================
        # 👉 여기를 확인하세요: 디버깅 지점
        # h와 g의 차원을 출력하여 문제의 원인을 직접 파악합니다.
        print("--- 디버깅 정보 ---")
        print(f"h shape: {h.shape}")
        print(f"g shape: {g.shape}")
        print("-------------------")
        # ======================================================================

        # h의 배치 크기에 맞게 g를 재구성합니다.
        if g.dim() == 1 or g.size(0) != h.size(0):
            g = g.view(h.size(0), -1)
        
        out = torch.cat([h, g], dim=1)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze()

# 6. ✨ SHAP 기반 MACCS 선택
rdkit_train, maccs_train = calc_rdkit_maccs(train['Canonical_Smiles'])
scaler = StandardScaler()
rdkit_scaled = scaler.fit_transform(rdkit_train)
selector = SelectFromModel(GradientBoostingRegressor(), max_features=30)
selector.fit(maccs_train, train['Inhibition'])
maccs_top = selector.transform(maccs_train)
X_global = np.concatenate([rdkit_scaled, maccs_top], axis=1)
y = train['Inhibition'].values

# 7. 🔁 Graph 변환
graph_list = []
for smi, g_feat in zip(train['Canonical_Smiles'], X_global):
    graph = mol_to_graph_data_obj(smi, g_feat)
    if graph:
        graph.y = torch.tensor([y[len(graph_list)]], dtype=torch.float)
        graph_list.append(graph)

# 8. 🎯 Optuna 최적화
def objective(trial):
    hidden = trial.suggest_categorical("hidden", [128, 256])
    layers = trial.suggest_int("layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch", [32, 64])
    patience = trial.suggest_int("patience", 10, 30)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(graph_list):
        train_dataset = [graph_list[i] for i in train_idx]
        val_dataset = [graph_list[i] for i in val_idx]
        model = GNNModel(5, X_global.shape[1], hidden, layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_score, best_epoch = 0, 0
        for epoch in range(1000):
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loader = DataLoader(val_dataset, batch_size=128)
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds += model(batch).cpu().numpy().tolist()
                    trues += batch.y.cpu().numpy().tolist()
            rmse, corr, score = compute_metrics(trues, preds)
            if score > best_score:
                best_score, best_epoch = score, epoch
            elif epoch - best_epoch > patience:
                break
        scores.append(best_score)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
joblib.dump(study, "./_save/optuna_study_gnn.pkl")

# 9. 📌 최적 파라미터로 전체 학습 후 테스트 예측
best_params = study.best_params
model = GNNModel(5, X_global.shape[1], best_params['hidden'], best_params['layers'], best_params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
train_loader = DataLoader(graph_list, batch_size=best_params['batch'], shuffle=True)
for epoch in range(300):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "./_save/best_gnn_model.pt")

# 10. 📤 제출 파일 생성
rdkit_test, maccs_test = calc_rdkit_maccs(test['Canonical_Smiles'])
rdkit_test_scaled = scaler.transform(rdkit_test)
maccs_test_top = selector.transform(maccs_test)
X_test_global = np.concatenate([rdkit_test_scaled, maccs_test_top], axis=1)

# 유효한 그래프 객체만 필터링
test_graphs = [mol_to_graph_data_obj(smi, feat) for smi, feat in zip(test['Canonical_Smiles'], X_test_global)]
test_graphs = [g for g in test_graphs if g is not None]

test_loader = DataLoader(test_graphs, batch_size=128)

model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds += model(batch).cpu().numpy().tolist()

submission['Predicted'] = preds
submission.to_csv('./_save/gnn_submission.csv', index=False)
print(f"Submission saved to ./_save/gnn_submission.csv")