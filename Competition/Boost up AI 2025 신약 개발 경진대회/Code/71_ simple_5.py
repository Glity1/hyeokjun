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
# import optuna # Optuna는 사용하지 않으므로 주석 처리 또는 삭제

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
submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

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

        self.fc1 = Linear(gat_out_channels_2 * num_heads_2 + global_feat_dim, mlp_hidden_dim)
        self.fc2 = Linear(mlp_hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_feat = data.global_feat.to(x.device)

        x = self.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.relu(self.bn2(self.gat2(x, edge_index)))

        x = global_mean_pool(x, batch)
        
        x = torch.cat([x, global_feat], dim=1)

        x = self.gnn_mlp_dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# Optuna 최적화 대신 최적의 하이퍼파라미터를 직접 적용
print("Optuna 최적화 대신 제공된 최적 하이퍼파라미터 적용...")

# 사용자가 제공한 최적의 하이퍼파라미터
best_params = {
    'gat_out_channels_1': 96, 
    'num_heads_1': 8, 
    'gat_out_channels_2': 192, 
    'num_heads_2': 8, 
    'gat_dropout_rate': 0.1, 
    'mlp_hidden_dim': 128, 
    'mlp_dropout_rate': 0.1, 
    'lr': 0.0018907552867128321, 
    'batch_size': 32
}

print(f"적용될 하이퍼파라미터: {best_params}")

# 최종 학습을 위한 데이터 로더
train_idx, val_idx = train_test_split(range(len(data_list)), test_size=0.2, random_state=seed)
train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=best_params['batch_size'])
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

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
final_scheduler = ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

best_final_score, best_final_rmse, best_final_corr = -np.inf, 0, 0
final_counter, final_patience = 20, 20 

print("\n최종 모델 학습 및 예측 시작...")
for epoch in range(1, 201):
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
    y_range = trues.max() - trues.min() # y_range 재계산
    
    # normalized_rmse 계산 방식 수정 및 y_range가 0인 경우 처리
    if y_range == 0:
        normalized_rmse = 0.0
    else:
        normalized_rmse = rmse / y_range
        
    score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

    print(f"[Final Epoch {epoch}] RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")

    final_scheduler.step(score)

    if score > best_final_score:
        best_final_score = score
        best_final_rmse = rmse
        best_final_corr = corr
        final_counter = 0
        torch.save(final_model.state_dict(), './_save/gnn_node_rdkit_maccs_fixed_params_model.pt') # 모델 저장명 변경
    else:
        final_counter += 1
        if final_counter >= final_patience:
            print("Final Training EarlyStopping")
            break

# 테스트 예측
final_model.load_state_dict(torch.load('./_save/gnn_node_rdkit_maccs_fixed_params_model.pt')) # 저장된 모델 로드
final_model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = final_model(batch)
        test_preds.append(out.cpu().numpy())
submit_df = submit_df_original.copy()
submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_node_rdkit_maccs_fixed_params_submission.csv', index=False) # 제출 파일명 변경

# 🎯 최종 결과 출력
print("\n🎯 [최종 결과 - 고정된 하이퍼파라미터 적용]")
print(f"Best Score : {best_final_score:.4f}")
print(f"Best RMSE  : {best_final_rmse:.4f}")
print(f"Best Corr  : {best_final_corr:.4f}")
print("\n✅ 제출 파일 저장 완료: ./_save/gnn_node_rdkit_maccs_fixed_params_submission.csv")