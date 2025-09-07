import os, platform, joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
import torch
from torch.nn import Linear, Module, Dropout, ModuleList, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import optuna

# 설정
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

# 점수 계산 함수
def calculate_score(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float, device=device)
    y_pred = torch.tensor(y_pred, dtype=torch.float, device=device)
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    denom = torch.max(y_true) - torch.min(y_true)
    A = rmse / denom if denom != 0 else torch.tensor(0.0, device=device)
    cov = torch.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    B = torch.clamp(cov / (y_true.std() * y_pred.std() + 1e-8), 0.0, 1.0)
    return (0.5 * (1 - torch.minimum(A, torch.tensor(1.0, device=device))) + 0.5 * B).item()

# 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
test = pd.read_csv('./_data/dacon/new/test.csv')
submit_df = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# MoleculeDataset 정의
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.has_target = 'Inhibition' in df.columns

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol = Chem.MolFromSmiles(row['Canonical_Smiles'])
        if mol is None: return None

        node_feats = [[
            a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
            int(a.GetHybridization()), int(a.GetIsAromatic()),
            a.GetNumExplicitHs(), a.GetNumImplicitHs(),
            a.GetIsotope(), a.GetNumRadicalElectrons(),
            a.GetChiralTag(), a.GetMass()
        ] for a in mol.GetAtoms()]
        x = torch.tensor(node_feats, dtype=torch.float)

        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge = [
                bond.GetBondTypeAsDouble(), int(bond.IsInRing()),
                int(bond.GetIsConjugated()), int(bond.GetStereo())
            ]
            edge_index += [[i, j], [j, i]]
            edge_attr += [edge, edge]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        desc = torch.tensor([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
            Descriptors.Chi0n(mol), Descriptors.Chi1n(mol),
            Descriptors.Chi0v(mol), Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol), Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol)
        ], dtype=torch.float)

        maccs = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        maccs_tensor = torch.tensor([float(b) for b in maccs], dtype=torch.float)
        z = torch.cat([desc, maccs_tensor]).unsqueeze(0)

        y = torch.tensor([row['Inhibition']], dtype=torch.float) if self.has_target else None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z)

    def __len__(self): return len(self.df)

# GNN 모델 정의 (GATv2 + GraphNorm + JK concat)
class GNN(Module):
    def __init__(self, node_input_dim, edge_input_dim, graph_input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.node_encoder = Linear(node_input_dim, hidden)
        self.edge_encoder = Linear(edge_input_dim, hidden)

        self.convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden, hidden, edge_dim=hidden))
            self.norms.append(GraphNorm(hidden))

        self.pool = global_mean_pool
        self.fc1 = Linear(hidden * (num_layers + 1) + graph_input_dim, hidden)
        self.fc2 = Linear(hidden, hidden // 2)
        self.fc3 = Linear(hidden // 2, 1)
        self.dropout = Dropout(dropout)
        self.act = ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = data.z.squeeze(1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch)
            x = self.act(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.pool(x, batch)
        x = torch.cat([x, z], dim=1)

        x = self.act(self.fc1(self.dropout(x)))
        x = self.act(self.fc2(self.dropout(x)))
        return self.fc3(x).view(-1)

# Optuna objective 함수
def objective(trial):
    params = {
        'seed': trial.suggest_int('seed', 1, 1000),
        'hidden': trial.suggest_categorical('hidden', [64, 128, 256]),
        'layers': trial.suggest_int('layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch': trial.suggest_categorical('batch', [32, 64, 128]),
        'patience': trial.suggest_int('patience', 10, 30)
    }

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    tr, vl = train_test_split(train, test_size=0.2, random_state=params['seed'])
    tr_loader = DataLoader(MoleculeDataset(tr), batch_size=params['batch'], shuffle=True)
    vl_loader = DataLoader(MoleculeDataset(vl), batch_size=params['batch'])

    sample = next(iter(tr_loader))
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    graph_dim = sample.z.shape[1]  # ✅ 수정된 부분

    model = GNN(node_dim, edge_dim, graph_dim,
                params['hidden'], params['layers'], params['dropout']).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    sch = ReduceLROnPlateau(opt, mode='max', patience=params['patience']//2)
    best, cnt = 0, 0

    for epoch in range(1, 201):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in vl_loader:
                batch = batch.to(device)
                y_true += batch.y.view(-1).tolist()
                y_pred += model(batch).tolist()
        score = calculate_score(y_true, y_pred)
        sch.step(score)
        if score > best: best = score; cnt = 0
        else: cnt += 1
        if cnt >= params['patience']: break
    return best

# Optuna 실행 및 로그 저장
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=60*60*2)
joblib.dump(study, './_save/optuna_gnnv2_study.pkl')

# 최적 파라미터 추출
best = study.best_trial.params
print(f"\nBest Score: {study.best_trial.value:.4f}")
print("Best Params:", best)

# 최종 학습 및 예측
np.random.seed(best['seed'])
torch.manual_seed(best['seed'])
torch.cuda.manual_seed_all(best['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_df, val_df = train_test_split(train, test_size=0.2, random_state=best['seed'])
train_loader = DataLoader(MoleculeDataset(train_df), batch_size=best['batch'], shuffle=True)
val_loader = DataLoader(MoleculeDataset(val_df), batch_size=best['batch'])

sample = next(iter(train_loader))
model = GNN(sample.x.shape[1], sample.edge_attr.shape[1], sample.z.shape[1],
            best['hidden'], best['layers'], best['dropout']).to(device)

opt = torch.optim.Adam(model.parameters(), lr=best['lr'])
sch = ReduceLROnPlateau(opt, mode='max', patience=best['patience']//2)
best_score = 0; counter = 0
model_path = './_save/gnn_optuna_best_model.pt'

for epoch in range(1, 501):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            y_true += batch.y.view(-1).tolist()
            y_pred += model(batch).tolist()
    score = calculate_score(y_true, y_pred)
    sch.step(score)
    print(f"[Epoch {epoch}] Val Score={score:.4f}")
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), model_path)
        counter = 0
    else:
        counter += 1
        if counter >= best['patience']:
            print("EarlyStopping")
            break

# 최종 모델 로드 및 예측
model.load_state_dict(torch.load(model_path))
model.eval()
test_loader = DataLoader(MoleculeDataset(test), batch_size=best['batch'])
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        test_preds += model(batch).cpu().tolist()

submit_df['Inhibition'] = np.clip(test_preds, 0, 100)
submit_path = f"./_save/submission_gnnv2_optuna_{best_score:.4f}.csv"
submit_df.to_csv(submit_path, index=False)
print(f"\n✅ Submission saved to: {submit_path}")
