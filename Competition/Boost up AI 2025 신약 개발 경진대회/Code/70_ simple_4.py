import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.AllChem import rdmolops
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import optuna # Optuna 라이브러리 임포트

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
submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv') # 최종 제출용 원본 저장

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
    
    # *** 중요 수정: global_feat를 (1, feature_dim) 형태로 만듭니다. ***
    global_feat = torch.tensor(global_feat, dtype=torch.float).unsqueeze(0) 
    
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

# GATv2 모델 클래스 (하이퍼파라미터 인자로 받도록 수정)
class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim, global_feat_dim,
                 gat_out_channels_1, num_heads_1, gat_dropout_rate,
                 gat_out_channels_2, num_heads_2,
                 mlp_hidden_dim, mlp_dropout_rate):
        super().__init__()
        self.gat1 = GATv2Conv(node_feat_dim, gat_out_channels_1, heads=num_heads_1, dropout=gat_dropout_rate)
        self.bn1 = BatchNorm1d(gat_out_channels_1 * num_heads_1)

        self.gat2 = GATv2Conv(gat_out_channels_1 * num_heads_1, gat_out_channels_2, heads=num_heads_2, dropout=gat_dropout_rate)
        self.bn2 = BatchNorm1d(gat_out_channels_2 * num_heads_2)

        self.relu = ReLU()
        self.gnn_mlp_dropout = Dropout(mlp_dropout_rate)

        # 최종 MLP 레이어
        self.fc1 = Linear(gat_out_channels_2 * num_heads_2 + global_feat_dim, mlp_hidden_dim)
        self.fc2 = Linear(mlp_hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_feat = data.global_feat.to(x.device) # global_feat는 이제 (batch_size, global_feat_dim) 형태

        # GATv2 레이어 적용
        x = self.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.relu(self.bn2(self.gat2(x, edge_index)))

        # 글로벌 평균 풀링
        x = global_mean_pool(x, batch)
        
        # *** 중요 수정: global_feat 차원 조정 로직 제거 ***
        # DataLoader가 이미 올바른 배치 형태로 global_feat를 쌓아주므로 불필요
        x = torch.cat([x, global_feat], dim=1)

        # MLP 레이어 적용
        x = self.gnn_mlp_dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# Optuna Objective 함수
def objective(trial):
    # 하이퍼파라미터 제안
    gat_out_channels_1 = trial.suggest_int('gat_out_channels_1', 32, 128, step=32)
    num_heads_1 = trial.suggest_categorical('num_heads_1', [2, 4, 8])
    gat_out_channels_2 = trial.suggest_int('gat_out_channels_2', 64, 256, step=32)
    num_heads_2 = trial.suggest_categorical('num_heads_2', [2, 4, 8])
    gat_dropout_rate = trial.suggest_float('gat_dropout_rate', 0.1, 0.5, step=0.1)

    mlp_hidden_dim = trial.suggest_int('mlp_hidden_dim', 64, 256, step=64)
    mlp_dropout_rate = trial.suggest_float('mlp_dropout_rate', 0.1, 0.5, step=0.1)

    # suggest_loguniform 대신 suggest_float(log=True) 사용
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) 
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    epochs = 100 # Optuna Trial당 최대 에폭 수 (조기 종료 적용)
    patience = 15 # 조기 종료를 위한 patience (Optuna Trial에서는 약간 짧게)


    # 데이터 로더 재생성 (Batch Size 변경 가능성 때문에)
    train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=batch_size)

    # 모델 인스턴스화
    model = GATv2WithGlobal(
        node_feat_dim=9, # 고정
        global_feat_dim=187, # 고정
        gat_out_channels_1=gat_out_channels_1,
        num_heads_1=num_heads_1,
        gat_dropout_rate=gat_dropout_rate,
        gat_out_channels_2=gat_out_channels_2,
        num_heads_2=num_heads_2,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout_rate=mlp_dropout_rate
    ).to(device)

    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()
    # 학습률 스케줄러 추가 (검증 점수 기준)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=False)

    best_score_trial = -np.inf
    counter = 0 # 조기 종료 카운터

    for epoch in range(1, epochs + 1):
        # 학습
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
        y_range = trues.max() - trues.min()
        normalized_rmse = rmse / y_range
        score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

        scheduler.step(score) # 스케줄러 업데이트

        # Optuna에 점수 보고 및 Pruning
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # 조기 종료 로직 (현재 Trial 내에서만 적용)
        if score > best_score_trial:
            best_score_trial = score
            counter = 0 # 최고 점수 갱신 시 카운터 초기화
        else:
            counter += 1
            if counter >= patience:
                break # 조기 종료

    return best_score_trial

# Optuna 스터디 생성 및 최적화
print("Optuna 최적화 시작...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed)) # seed 추가
study.optimize(objective, n_trials=50, timeout=3600) # 예시: 50번의 시도 또는 1시간 타임아웃

print("\nOptuna 최적화 완료!")
print(f"최적의 Trial 값 (Best Trial Score): {study.best_trial.value:.4f}")
print(f"최적의 하이퍼파라미터: {study.best_trial.params}")

# Optuna로 찾은 최적의 하이퍼파라미터로 최종 모델 학습 및 예측
print("\n최적 하이퍼파라미터로 최종 모델 학습 및 예측 시작...")
best_params = study.best_trial.params

# 최종 학습을 위한 데이터 로더 (Optuna Trial과 동일한 분할 사용)
train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=best_params['batch_size'])
test_loader = DataLoader(test_data, batch_size=best_params['batch_size']) # 테스트 로더도 최적 배치 크기 사용

final_model = GATv2WithGlobal(
    node_feat_dim=9,
    global_feat_dim=187,
    gat_out_channels_1=best_params['gat_out_channels_1'],
    num_heads_1=best_params['num_heads_1'],
    gat_dropout_rate=best_params['gat_dropout_rate'],
    gat_out_channels_2=best_params['gat_out_channels_2'],
    num_heads_2=best_params['num_heads_2'],
    mlp_hidden_dim=best_params['mlp_hidden_dim'],
    mlp_dropout_rate=best_params['mlp_dropout_rate']
).to(device)

final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
final_loss_fn = torch.nn.SmoothL1Loss()
final_scheduler = ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True) # verbose=True로 변경

best_final_score, best_final_rmse, best_final_corr = -np.inf, 0, 0
final_counter, final_patience = 20, 20 # 최종 학습에서는 patience를 좀 더 길게 설정할 수 있음

for epoch in range(1, 201): # 최종 학습 에폭은 더 길게 설정
    final_model.train()
    for batch in train_loader:
        batch = batch.to(device)
        final_optimizer.zero_grad()
        out = final_model(batch)
        loss = final_loss_fn(out, batch.y.view(-1))
        loss.backward()
        final_optimizer.step()

    final_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = final_model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    corr = pearsonr(trues, preds)[0]
    std = np.std(trues)
    normalized_rmse = rmse / std if std != 0 else 0 
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

    print(f"[Final Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")

    final_scheduler.step(score) # 최종 스케줄러 업데이트

    if score > best_final_score:
        best_final_score = score
        best_final_rmse = rmse
        best_final_corr = corr
        final_counter = 0
        torch.save(final_model.state_dict(), './_save/op_gnn_node_rdkit_maccs_optuna_best_model.pt')
    else:
        final_counter += 1
        if final_counter >= final_patience:
            print("Final Training EarlyStopping")
            break

# 테스트 예측
final_model.load_state_dict(torch.load('./_save/op_gnn_node_rdkit_maccs_optuna_best_model.pt'))
final_model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = final_model(batch)
        test_preds.append(out.cpu().numpy())
submit_df = submit_df_original.copy() # 원본 제출 양식 사용
submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/op_gnn_node_rdkit_maccs_optuna_submission.csv', index=False)

# 🎯 최종 결과 출력
print("\n🎯 [최종 결과 - Optuna로 찾은 Best Hyperparameters 기준]")
print(f"Best Score : {best_final_score:.4f}")
print(f"Best RMSE  : {best_final_rmse:.4f}")
print(f"Best Corr  : {best_final_corr:.4f}")
print("\n✅ 제출 파일 저장 완료: ./_save/op_gnn_node_rdkit_maccs_optuna_submission.csv")