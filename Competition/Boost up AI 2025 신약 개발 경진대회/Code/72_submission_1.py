import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, GraphNorm

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split

import optuna

SEED = 30
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'outputs', 'submissions')

# -------------------------
# 1) Features & Data Utils
# -------------------------

def smile_to_features(smile: str, y: float=None):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    # 1. Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),          # 원자 번호
            atom.GetDegree(),             # 결합된 원자 수
            atom.GetFormalCharge(),       # 형식 전하
            int(atom.GetHybridization()), # 혼성화 상태
            int(atom.GetIsAromatic()),    # 방향족 여부
            atom.GetNumExplicitHs(),      # 명시적 수소 수
            atom.GetNumImplicitHs(),      # 암시적 수소 수
            atom.GetMass(),               # 질량
            atom.GetIsotope(),            # 동위원소 정보
            atom.GetNumRadicalElectrons(),# 라디칼 전자 수

        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # 2. Edge features & index
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_feats = [
            bond.GetBondTypeAsDouble(),   # 결합 유형
            int(bond.IsInRing()),         # 고리 내 여부
            int(bond.GetIsConjugated()),  # 공액 결합 여부
            int(bond.GetStereo()),        # 스테레오 정보
        ]
        edge_attr += [edge_feats, edge_feats]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 3. Graph-Level features
    graph_feat = [
        Descriptors.MolWt(mol),              # 분자량
        Descriptors.MolLogP(mol),            # 소수성
        Descriptors.NumHDonors(mol),         # 수소 결합 donor 수
        Descriptors.NumHAcceptors(mol),      # 수소 결합 acceptor 수
        Descriptors.TPSA(mol),               # 극성 표면적
        Descriptors.NumRotatableBonds(mol),  # 회전 가능한 결합 수
        Descriptors.RingCount(mol),          # 고리 수
        Descriptors.HeavyAtomCount(mol),     # 수소를 제외한 원자 수
        Descriptors.NumAromaticRings(mol),   # 방향족 고리 수
    ]

    graph_feat = torch.tensor(graph_feat, dtype=torch.float).unsqueeze(0)

    y_tensor = torch.tensor([y], dtype=torch.float) if y is not None else None

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        graph_feat=graph_feat
    )

def calculate_score(y_true, y_pred):
    # RMSE 계산
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    denominator = torch.max(y_true) - torch.min(y_true)
    A = rmse / denominator if denominator != 0 else torch.tensor(0.0, device=y_true.device)

    # Pearson 상관계수 계산
    y_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    cov = torch.mean((y_true - y_mean) * (y_pred - y_pred_mean))
    std_y = torch.std(y_true)
    std_y_hat = torch.std(y_pred)
    if std_y * std_y_hat == 0:
        B = torch.tensor(0.0, device=y_true.device)
    else:
        pcc = cov / (std_y * std_y_hat)
        B = torch.clamp(pcc, 0.0, 1.0)

    score = 0.5 * (1 - torch.minimum(A, torch.tensor(1.0, device=y_true.device))) + 0.5 * B
    return score.item()  # float로 반환

# -------------------------
# 2) Unified Dataset
# -------------------------

class MoleculeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.has_target = 'Inhibition' in df.columns
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Canonical_Smiles']
        inhibition = row['Inhibition'] if self.has_target else None
        return smile_to_features(smiles, y=inhibition)

# -------------------------
# 3) GINE Regressor Model
# -------------------------

class GINERegressionModel(nn.Module):
    def __init__(self, 
                 node_input_dim: int, 
                 edge_input_dim: int, 
                 graph_feat_dim: int, 
                 hidden_dim: int, 
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()

        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_fn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_fn)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))

        self.pool = global_mean_pool

        # MLP head: [GNN_out + graph_feat_dim] → hidden → 1
        jk_dim = hidden_dim * (num_layers + 1)
        total_input_dim = jk_dim + graph_feat_dim

        self.mlp = MLPHead(
            input_dim = total_input_dim, 
            hidden_dim = hidden_dim, 
            dropout = dropout
        )

    def forward(self, x, edge_index, edge_attr, batch, graph_feat):
        h = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        h_list = [h]
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h, batch)
            h = F.relu(h)
            h = h + h_in
            h_list.append(h)
        h_jk = torch.cat(h_list, dim=1)

        h_graph = self.pool(h_jk, batch)
        h_total = torch.cat([h_graph, graph_feat], dim=1)
        out = self.mlp(h_total)  # [batch_size, 1]

        return out.view(-1)  # [batch_size]

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1)  # 회귀값
        )

    def forward(self, x):
        return self.layers(x).view(-1)

# -------------------------------
# 4) Utility Functions
# -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------
# 5) 데이터 로드
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'outputs', 'submissions')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
predict_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


# -------------------------------
# 6) 하이퍼 파라미터 튜닝
# -------------------------------
def objective(trial):
    # 하이퍼파라미터 샘플링
    seed = trial.suggest_int('seed', 1, 1000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 100, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = round(trial.suggest_float('dropout', 0.0, 0.5), 4)
    lr = round(trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True), 4)

    set_seed(seed)

    # train/val split
    tr_df, vl_df = train_test_split(train_df, test_size=0.2, random_state=seed)

    tr_ds = MoleculeDataset(tr_df)
    vl_ds = MoleculeDataset(vl_df)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    sample_batch = next(iter(tr_loader))
    node_dim = sample_batch.x.shape[1]
    edge_dim = sample_batch.edge_attr.shape[1]
    graph_dim = sample_batch.graph_feat.shape[1]

    model = GINERegressionModel(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graph_feat_dim=graph_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.7)

    best_val_score = 0.0
    patience_cnt = 0
    max_epochs = 50

    for _ in range(1, max_epochs + 1):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            loss = F.mse_loss(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch in vl_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
                val_true.extend(batch.y.view(-1).tolist())
                val_pred.extend(preds.tolist())
        val_score = calculate_score(torch.tensor(val_true), torch.tensor(val_pred))
        scheduler.step(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                break

    return best_val_score

# Study 설정 및 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=60*60*2)

best_params = study.best_params
print('Best hyperparameters:', best_params)

# -------------------------------
# 7) 최종 학습 및 예측
# -------------------------------
# best_params를 사용하여 Seed 설정
set_seed(best_params['seed'])

# Train+Val 결합, Test 분리 (0.2)
final_df, test_df = train_test_split(train_df, test_size=0.2, random_state=best_params['seed'])
final_ds = MoleculeDataset(final_df)
test_ds = MoleculeDataset(test_df)
final_loader = DataLoader(final_ds, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)

# predict_loader 재생성
predict_ds = MoleculeDataset(predict_df)
predict_loader = DataLoader(predict_ds, batch_size=best_params['batch_size'], shuffle=False)

# 모델 초기화
sample_batch = next(iter(final_loader))
node_dim = sample_batch.x.shape[1]
edge_dim = sample_batch.edge_attr.shape[1]
graph_dim = sample_batch.graph_feat.shape[1]

model = GINERegressionModel(
    node_input_dim=node_dim,
    edge_input_dim=edge_dim,
    graph_feat_dim=graph_dim,
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout']
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.7)

# Early stopping 변수 초기화
best_val = 0.0
cnt = 0
for epoch in range(1, 201):
    model.train()
    for batch in final_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
        loss = F.mse_loss(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            y_true.extend(batch.y.view(-1).tolist())
            y_pred.extend(preds.tolist())
    val_score = calculate_score(torch.tensor(y_true), torch.tensor(y_pred))
    scheduler.step(val_score)
    print(f'[Epoch {epoch:03d}] Val Score={val_score:.4f}')

    if val_score > best_val:
        best_val = val_score
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'optimized_gine_model_01.pt'))
        cnt = 0
    else:
        cnt += 1
        if cnt >= 50:
            print(f'Early stopping at epoch {epoch}')
            break

# 최종 모델 로드 및 테스트 평가
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'optimized_gine_model_01.pt')))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
        y_true.extend(batch.y.view(-1).tolist())
        y_pred.extend(preds.tolist())

test_score = calculate_score(torch.tensor(y_true), torch.tensor(y_pred))
print(f"\nFinal Test Score: {test_score:.4f}")

# 예측 및 제출 파일 생성
predictions = []
with torch.no_grad():
    for batch in predict_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
        predictions.extend(preds.tolist())

submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
submission['Inhibition'] = predictions
submission_path = os.path.join(RESULT_DIR, 'optimized_gine.csv')
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

'''
Best hyperparameters: {'seed': 153, 'batch_size': 32, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.461216060067939, 'learning_rate': 0.00039927949462805043}
Final Test Score: 0.6262
'''