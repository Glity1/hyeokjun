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

import optuna

# ----------------------------------------------------
#               설정 및 데이터 전처리
# ----------------------------------------------------

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
try:
    train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')
except FileNotFoundError:
    print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

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
    if mol is None: return None
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
data_list = [smiles_to_graph_with_global(s, l) for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train), desc="Processing train data")]
data_list = [d for d in data_list if d is not None]
test_data = [smiles_to_graph_with_global(s) for s in tqdm(smiles_test, desc="Processing test data")]
test_data = [d for d in test_data if d is not None]

# ----------------------------------------------------
#                     모델 정의
# ----------------------------------------------------

class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, global_feat_dim=187, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm1d(hidden_channels * heads)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

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

# ----------------------------------------------------
#                    Optuna Objective 함수
# ----------------------------------------------------

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])

    # 모델, 옵티마이저, 손실 함수 초기화
    model = GATv2WithGlobal(
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    # 데이터 로더
    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=batch_size)

    # 학습 루프
    best_score = -np.inf
    patience_counter = 0
    patience = 10 # Optuna에서는 적절한 조기 종료 설정을 통해 빠르게 탐색

    for epoch in range(1, 101):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        # 검증
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

        # Optuna에 중간 결과 전달 (가지치기)
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # 조기 종료 로직
        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_score

# ----------------------------------------------------
#                    Optuna Study 실행
# ----------------------------------------------------

# Optuna Study 생성
# 'maximize' 방향으로 최적화 (스코어를 높이는 방향)
study = optuna.create_study(direction='maximize')
# 최적화 실행. n_trials는 시도할 하이퍼파라미터 조합의 수
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n\n--- Optuna 최적화 결과 ---")
print("최적 하이퍼파라미터:", study.best_params)
print("최고 Score:", study.best_value)

# ----------------------------------------------------
#               최적 하이퍼파라미터로 최종 학습
# ----------------------------------------------------

best_params = study.best_params

# 최적의 하이퍼파라미터로 최종 모델 학습
print("\n--- 최적의 하이퍼파라미터로 최종 모델 학습 시작 ---")
model = GATv2WithGlobal(
    hidden_channels=best_params['hidden_channels'],
    heads=best_params['heads'],
    dropout=best_params['dropout']
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn = torch.nn.SmoothL1Loss()
train_loader = DataLoader(data_list, batch_size=best_params['batch_size'], shuffle=True) # 전체 데이터를 사용
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

best_score, best_rmse, best_corr = -np.inf, 0, 0
counter, patience = 0, 20

# 최종 학습 루프
for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
    
    # 여기서는 전체 데이터로 학습하므로, 학습 후 모델 저장만 진행
    # 더 정확한 검증을 원할 경우 train-val 분할을 다시 수행해야 함
    
    # 20 에포크마다 모델 저장
    if (epoch % 20 == 0) or (epoch == 100):
        print(f"Saving model at epoch {epoch}")
        torch.save(model.state_dict(), f'./_save/gnn_optuna_final_epoch_{epoch}.pt')

# ----------------------------------------------------
#                     최종 예측 및 저장
# ----------------------------------------------------

# 가장 마지막에 저장된 모델을 불러와서 예측
final_model_path = './_save/gnn_optuna_final_epoch_100.pt'
model.load_state_dict(torch.load(final_model_path))
model.eval()

test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())

submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_optuna_submission.csv', index=False)

print("\n✅ 최종 예측 완료: ./_save/gnn_optuna_submission.csv")