import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Dropout, BatchNorm1d # BatchNorm1d 추가
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
import warnings

# RDKit Deprecation 경고를 무시합니다.
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
    # NaN이 포함된 예측값을 처리합니다.
    y_pred_filtered = np.array(y_pred)
    y_true_filtered = np.array(y_true)
    
    # NaN이 아닌 인덱스를 찾습니다.
    not_nan_indices = ~np.isnan(y_pred_filtered)
    
    if not np.any(not_nan_indices): # 모든 예측값이 NaN인 경우
        return 1.0, 0.0, 0.0 # RMSE를 최대값으로, 상관계수와 스코어를 0으로 반환하여 Optuna가 이 Trial을 피하게 함
    
    y_true_clean = y_true_filtered[not_nan_indices]
    y_pred_clean = y_pred_filtered[not_nan_indices]
    
    if len(y_true_clean) == 0: # 유효한 값이 없는 경우
        return 1.0, 0.0, 0.0

    rmse = mean_squared_error(y_true_clean, y_pred_clean, squared=False)
    # y_true_clean에 단일 값만 있는 경우 pearsonr 오류 방지
    if len(np.unique(y_true_clean)) == 1 or len(np.unique(y_pred_clean)) == 1:
        corr = 0.0 # 상관계수 계산 불가
    else:
        corr = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        if np.isnan(corr): # 상관계수가 NaN이 되는 경우 (예: 모든 예측값이 같을 때)
            corr = 0.0

    # 데이터의 스케일로 RMSE를 정규화하여 score 계산 시 더 합리적인 값을 얻도록 합니다.
    # train['Inhibition']의 전체 스케일을 사용하여 정규화
    data_range = train['Inhibition'].max() - train['Inhibition'].min()
    if data_range == 0: # 데이터 범위가 0인 경우 (모든 값이 동일)
        normalized_rmse = 0.0
    else:
        normalized_rmse = rmse / data_range
        
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * max(corr, 0)
    return rmse, corr, score

# 5. 🧠 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, in_dim, global_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        # 입력 레이어
        self.convs.append(GATv2Conv(in_dim, hidden))
        self.norms.append(GraphNorm(hidden)) # GraphNorm은 Node 레벨에서 작동하므로 hidden 차원
        
        # 중간 레이어
        for _ in range(layers - 1):
            self.convs.append(GATv2Conv(hidden, hidden))
            self.norms.append(GraphNorm(hidden))

        self.jump = JumpingKnowledge("cat") # 각 레이어의 풀링된 출력을 concat
        
        # 마지막 Linear 레이어
        self.fc1 = Linear(hidden * layers + global_dim, hidden)
        self.fc2 = Linear(hidden, 1)
        
        self.dropout = Dropout(dropout)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        outs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            # GATv2Conv 후에 배치 정규화 (GraphNorm) 및 ReLU 적용
            x = self.relu(norm(x)) 
            
            # 각 conv 레이어의 출력을 풀링하여 저장
            outs.append(global_mean_pool(x, batch))
        
        # 모든 레이어의 풀링된 출력을 결합
        h = torch.cat(outs, dim=1)
        
        # Global Feature 가져오기
        g = data.global_feat.to(h.device)

        # g의 배치 크기를 h와 일치시키기 (g가 스칼라로 들어올 경우를 대비)
        if g.dim() == 1 or g.size(0) != h.size(0):
            g = g.view(h.size(0), -1)
        
        # GNN 출력과 Global Feature를 결합
        out = torch.cat([h, g], dim=1)
        
        # 최종 FC 레이어
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)

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
    # 학습률 범위를 더 낮게 조정
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) 
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
        
        for epoch in range(1000): # 에폭 수 고정
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred.squeeze(), batch.y)
                
                # 그래디언트 클리핑 추가 (옵션, 필요시 주석 해제)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loader = DataLoader(val_dataset, batch_size=128)
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_batch = model(batch)
                    preds += pred_batch.squeeze().cpu().numpy().tolist()
                    trues += batch.y.cpu().numpy().tolist()
            
            # compute_metrics 함수에서 NaN 처리 로직이 추가되었으므로, 여기서 None을 반환하지 않음
            rmse, corr, score = compute_metrics(trues, preds)
            
            # score가 NaN이면 Optuna에 전달되지 않도록 처리
            if np.isnan(score):
                return None # Optuna가 이 시도를 실패로 간주하도록 함
            
            if score > best_score:
                best_score, best_epoch = score, epoch
            elif epoch - best_epoch > patience:
                break
        scores.append(best_score)
    
    # K-fold 교차 검증 중 하나라도 NaN이 발생하여 None이 반환되는 경우를 대비
    if any(np.isnan(s) for s in scores):
        return None

    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # n_trials를 30으로 설정
joblib.dump(study, "./_save/optuna_study_gnn.pkl")

# 9. 📌 최적 파라미터로 전체 학습 후 테스트 예측
best_params = study.best_params # Optuna 스터디에서 best_trial이 없으면 여기서 에러가 나므로 주의
model = GNNModel(5, X_global.shape[1], best_params['hidden'], best_params['layers'], best_params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
train_loader = DataLoader(graph_list, batch_size=best_params['batch'], shuffle=True)

# 최종 모델 학습 (에폭 수를 줄이거나, 스케줄러를 적용하는 것을 고려)
for epoch in range(300): # 예시 에폭 수
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred.squeeze(), batch.y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "./_save/best_gnn_model.pt")

# 10. 📤 제출 파일 생성
rdkit_test, maccs_test = calc_rdkit_maccs(test['Canonical_Smiles'])
scaler_test = StandardScaler() # 테스트 데이터는 훈련 데이터의 scaler로 transform만 해야 합니다.
rdkit_test_scaled = scaler.fit_transform(rdkit_test) # 이 부분을 수정해야 합니다. 훈련 데이터로 fit된 scaler 사용
# 정확한 스케일러 사용: rdkit_scaled = scaler.fit_transform(rdkit_train) 에서 fit된 scaler를 사용해야 합니다.
# 이미 위에 scaler.fit_transform(rdkit_train)이 있으므로, 여기서는 transform만 합니다.
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
        preds += model(batch).squeeze().cpu().numpy().tolist()

submission['Predicted'] = preds
submission.to_csv('./_save/gnn_submission.csv', index=False)
print(f"Submission saved to ./_save/gnn_submission.csv")