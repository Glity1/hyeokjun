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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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

# ----------------------------------------------------
#               엣지(결합) 피처 정의 (여러 버전) - 추천 조합 추가됨
# ----------------------------------------------------
def bond_features_basic(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.BondType.SINGLE), # 단일 결합
        int(bt == Chem.BondType.DOUBLE), # 이중 결합
        int(bt == Chem.BondType.TRIPLE), # 삼중 결합
        int(bt == Chem.BondType.AROMATIC), # 방향족 결합
        int(bond.GetIsConjugated()), # 공액 여부
        int(bond.IsInRing()), # 고리 내 결합 여부
        int(bond.GetStereo() == Chem.BondStereo.STEREOANY), # 스테레오화학 정보
        int(bond.GetStereo() == Chem.BondStereo.STEREOCIS),
        int(bond.GetStereo() == Chem.BondStereo.STEREOTRANS)
    ]

def bond_features_all_extended(bond):
    # 기본 결합 특성
    features = bond_features_basic(bond)
    
    # 추가: 인접 원자의 일부 정보
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    # 원자 번호
    features.append(begin_atom.GetAtomicNum())
    features.append(end_atom.GetAtomicNum())
    
    # 전기음성도
    features.append(electronegativity.get(begin_atom.GetAtomicNum(), 0))
    features.append(electronegativity.get(end_atom.GetAtomicNum(), 0))
    
    # 형식 전하
    features.append(begin_atom.GetFormalCharge())
    features.append(end_atom.GetFormalCharge())
    
    # 혼성화 상태 (정수형)
    features.append(int(begin_atom.GetHybridization()))
    features.append(int(end_atom.GetHybridization()))

    return features

# 추천 조합 1: 스테레오 정보 강화
def bond_features_extended_stereo(bond):
    bt = bond.GetBondType()
    features = [
        int(bt == Chem.BondType.SINGLE),
        int(bt == Chem.BondType.DOUBLE),
        int(bt == Chem.BondType.TRIPLE),
        int(bt == Chem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo() == Chem.BondStereo.STEREOANY),
        int(bond.GetStereo() == Chem.BondStereo.STEREOCIS),
        int(bond.GetStereo() == Chem.BondStereo.STEREOTRANS),
        int(bond.GetStereo() == Chem.BondStereo.STEREOE), # E/Z 스테레오
        int(bond.GetStereo() == Chem.BondStereo.STEREOZ),
        int(bond.GetStereo() == Chem.BondStereo.STEREONONE), # 스테레오 정보 없음
    ]
    return features

# 추천 조합 2: 인접 원자의 혼성화 및 원자 번호 정보
def bond_features_hybridization(bond):
    features = bond_features_basic(bond)
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    features.append(int(begin_atom.GetHybridization()))
    features.append(int(end_atom.GetHybridization()))
    features.append(begin_atom.GetAtomicNum())
    features.append(end_atom.GetAtomicNum())
    return features

# 추천 조합 3: 결합이 속한 고리 정보
def bond_features_ring_info(bond):
    features = bond_features_basic(bond)
    features.append(int(bond.IsInRingSize(3)))
    features.append(int(bond.IsInRingSize(4)))
    features.append(int(bond.IsInRingSize(5)))
    features.append(int(bond.IsInRingSize(6)))
    features.append(int(bond.IsInRingSize(7)))
    return features

# 추천 조합 4: 인접 원자의 결합 차수 정보
def bond_features_degree_neighbors(bond):
    features = bond_features_basic(bond)
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    features.append(begin_atom.GetDegree())
    features.append(end_atom.GetDegree())
    features.append(begin_atom.GetTotalNumHs())
    features.append(end_atom.GetTotalNumHs())
    return features

# ----------------------------------------------------
#               MACCS 키 중요도 분석
# ----------------------------------------------------

def get_only_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return maccs_arr.astype(np.float32)
    except Exception:
        return None

print("MACCS 키 중요도 분석을 위한 데이터 추출 중...")
maccs_data_list = []
valid_y = []
valid_smiles = []

for i, s in enumerate(tqdm(smiles_train, desc="Extracting all MACCS keys")):
    maccs_feat = get_only_maccs(s)
    if maccs_feat is not None:
        maccs_data_list.append(maccs_feat)
        valid_y.append(y[i])
        valid_smiles.append(s)

maccs_data = np.array(maccs_data_list)
valid_y = np.array(valid_y)

if maccs_data.shape[0] == 0:
    print("유효한 MACCS 키 데이터가 없습니다. 종료합니다.")
    exit()

print(f"랜덤 포레스트 모델 학습 중... (MACCS 데이터 shape: {maccs_data.shape})")
rf_model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
rf_model.fit(maccs_data, valid_y)

feature_importances = rf_model.feature_importances_
maccs_importance_dict = {i: importance for i, importance in enumerate(feature_importances)}
sorted_maccs_by_importance = sorted(maccs_importance_dict.items(), key=lambda item: item[1], reverse=True)

top_30_maccs_indices = [idx for idx, _ in sorted_maccs_by_importance[:30]]
top_30_maccs_indices.sort()
print(f"\n✅ 랜덤 포레스트 중요도로 선택된 Top 30 MACCS Key 인덱스 (0-indexed): {top_30_maccs_indices}")

# ----------------------------------------------------
#               전처리 함수 재정의 및 데이터셋 생성
# ----------------------------------------------------

def get_rdkit_and_selected_maccs(smiles, selected_maccs_indices):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        desc = [f(mol) for f in rdkit_desc_list]
        desc = np.nan_to_num(desc).astype(np.float32)
        
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

        selected_maccs_arr = maccs_arr[selected_maccs_indices].astype(np.float32)
        
        return np.concatenate([desc, selected_maccs_arr])
    except Exception:
        return None

def smiles_to_graph_final(smiles, label=None, selected_maccs_indices=None, scaler=None, bond_feature_extractor=None, edge_feature_dim=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    global_feat = get_rdkit_and_selected_maccs(smiles, selected_maccs_indices)
    
    global_feat_dim_expected = 20 + len(selected_maccs_indices)
    if global_feat is None or len(global_feat) != global_feat_dim_expected:
        return None
    
    global_feat_scaled = scaler.transform(global_feat.reshape(1, -1)).flatten()
    
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_indices.append([u, v])
        edge_indices.append([v, u])
        bf = bond_feature_extractor(bond) 
        edge_attrs.append(bf)
        edge_attrs.append(bf)
    
    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    global_feat_tensor = torch.tensor(global_feat_scaled, dtype=torch.float).view(-1)
    
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, global_feat=global_feat_tensor, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, edge_attr=edge_attr)

global_features_all = [
    get_rdkit_and_selected_maccs(s, top_30_maccs_indices) 
    for s in smiles_train
    if get_rdkit_and_selected_maccs(s, top_30_maccs_indices) is not None
]
global_features_all = np.array(global_features_all)
scaler = StandardScaler()
scaler.fit(global_features_all)

# ----------------------------------------------------
#               모델 정의
# ----------------------------------------------------
global_feat_dim_updated = 20 + len(top_30_maccs_indices)

class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim=9, edge_feat_dim=None, global_feat_dim=global_feat_dim_updated, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        
        if edge_feat_dim is None:
             raise ValueError("edge_feat_dim must be provided and not None.")
             
        self.gat1 = GATv2Conv(node_feat_dim, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_feat_dim)
        self.bn2 = BatchNorm1d(hidden_channels * heads)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        
        self.fc1 = Linear(hidden_channels * heads + global_feat_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        global_feat = data.global_feat.to(x.device)
        
        x = self.relu(self.bn1(self.gat1(x, edge_index, edge_attr)))
        x = self.relu(self.bn2(self.gat2(x, edge_index, edge_attr)))
        
        x = global_mean_pool(x, batch)
        
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
        
        if global_feat.size(0) != x.size(0):
             global_feat = global_feat.view(x.size(0), -1)

        x = torch.cat([x, global_feat], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# ----------------------------------------------------
#               Optuna Objective 함수 - 엣지 피처 선택 로직 확장
# ----------------------------------------------------

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5) 
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])

    # 엣지 피처 조합 선택 (여기 새로운 옵션들을 추가)
    edge_feature_choice = trial.suggest_categorical(
        'edge_feature_choice', 
        ['basic', 'all_extended', 'extended_stereo', 'hybridization_info', 'ring_info', 'degree_neighbors']
    )

    current_bond_feature_extractor = None
    current_edge_feature_dim = None
    
    # 엣지 피처 함수와 차원 매핑
    bond_feature_map = {
        'basic': (bond_features_basic, 9),
        'all_extended': (bond_features_all_extended, 17),
        'extended_stereo': (bond_features_extended_stereo, 12),
        'hybridization_info': (bond_features_hybridization, 13),
        'ring_info': (bond_features_ring_info, 14),
        'degree_neighbors': (bond_features_degree_neighbors, 13)
    }
    
    # 안전 장치: 'CC' 분자로 차원 계산하여 기본값 대체
    try:
        # 'extended_stereo'와 'ring_info'는 스테레오 및 링이 있는 분자로 차원 계산
        dim_extended_stereo = len(bond_features_extended_stereo(Chem.MolFromSmiles('C/C=C/C').GetBonds()[1]))
        dim_ring_info = len(bond_features_ring_info(Chem.MolFromSmiles('C1CC1').GetBonds()[0]))
        
        bond_feature_map['extended_stereo'] = (bond_features_extended_stereo, dim_extended_stereo)
        bond_feature_map['ring_info'] = (bond_features_ring_info, dim_ring_info)
    except Exception as e:
        print(f"Warning: Could not determine dynamic bond feature dimensions for some types. Using defaults. Error: {e}")

    current_bond_feature_extractor, current_edge_feature_dim = bond_feature_map[edge_feature_choice]
    
    print(f"\nOptuna Trial {trial.number}: Using edge feature choice '{edge_feature_choice}' (dim: {current_edge_feature_dim})")

    data_list_trial = [smiles_to_graph_final(s, l, top_30_maccs_indices, scaler, 
                                            current_bond_feature_extractor, current_edge_feature_dim) 
                       for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train), desc=f"Processing train data for trial {trial.number}")]
    data_list_trial = [d for d in data_list_trial if d is not None]

    if not data_list_trial:
        print(f"Warning: No valid data generated for trial {trial.number}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    model = GATv2WithGlobal(
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        edge_feat_dim=current_edge_feature_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    train_idx, val_idx = train_test_split(range(len(data_list_trial)), test_size=0.2, random_state=seed)
    train_loader = DataLoader([data_list_trial[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([data_list_trial[i] for i in val_idx], batch_size=batch_size)

    best_score = -np.inf
    patience_counter = 0
    patience = 10

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
        
        y_range = trues.max() - trues.min()
        normalized_rmse = rmse / y_range if y_range > 0 else 1.0
        score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_score

# ----------------------------------------------------
#               Optuna Study 실행
# ----------------------------------------------------

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n\n--- Optuna 최적화 결과 ---")
print("최적 하이퍼파라미터:", study.best_params)
print("최고 Score:", study.best_value)

# ----------------------------------------------------
#               최적 하이퍼파라미터로 최종 학습
# ----------------------------------------------------

best_params = study.best_params

final_edge_feature_choice = best_params['edge_feature_choice']

# 엣지 피처 함수와 차원 매핑을 다시 정의
bond_feature_map = {
    'basic': bond_features_basic,
    'all_extended': bond_features_all_extended,
    'extended_stereo': bond_features_extended_stereo,
    'hybridization_info': bond_features_hybridization,
    'ring_info': bond_features_ring_info,
    'degree_neighbors': bond_features_degree_neighbors
}

final_bond_feature_extractor = bond_feature_map.get(final_edge_feature_choice, bond_features_all_extended)

# 최종 엣지 피처 차원 계산
try:
    if final_edge_feature_choice in ['extended_stereo', 'ring_info']:
        # 스테레오 및 링 정보가 있는 분자로 차원 계산
        test_mol = Chem.MolFromSmiles('C/C=C/C' if final_edge_feature_choice == 'extended_stereo' else 'C1CC1')
        test_bond = test_mol.GetBonds()[0]
        final_edge_feature_dim = len(final_bond_feature_extractor(test_bond))
    else:
        final_edge_feature_dim = len(final_bond_feature_extractor(Chem.MolFromSmiles('CC').GetBonds()[0]))
except Exception:
    # 예외 발생 시 기본값으로 설정
    print(f"Warning: Could not determine final bond feature dimension for '{final_edge_feature_choice}'. Using default (17).")
    final_edge_feature_dim = 17


print(f"\n--- 최적의 하이퍼파라미터 및 엣지 피처 ({final_edge_feature_choice}, Dim: {final_edge_feature_dim})로 최종 모델 학습 시작 ---")

final_data_list = [smiles_to_graph_final(s, l, top_30_maccs_indices, scaler, 
                                         final_bond_feature_extractor, final_edge_feature_dim) 
                   for s, l in tqdm(zip(smiles_train, y), total=len(smiles_train), desc="Processing final train data")]
final_data_list = [d for d in final_data_list if d is not None]

final_test_data = [smiles_to_graph_final(s, None, top_30_maccs_indices, scaler, 
                                         final_bond_feature_extractor, final_edge_feature_dim) 
                   for s in tqdm(smiles_test, desc="Processing final test data")]
final_test_data = [d for d in final_test_data if d is not None]

model = GATv2WithGlobal(
    hidden_channels=best_params['hidden_channels'],
    heads=best_params['heads'],
    dropout=best_params['dropout'],
    edge_feat_dim=final_edge_feature_dim
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn = torch.nn.SmoothL1Loss()

train_idx, val_idx = train_test_split(range(len(final_data_list)), test_size=0.2, random_state=seed)
final_train_loader = DataLoader([final_data_list[i] for i in train_idx], batch_size=best_params['batch_size'], shuffle=True)
final_val_loader = DataLoader([final_data_list[i] for i in val_idx], batch_size=best_params['batch_size'])
test_loader = DataLoader(final_test_data, batch_size=best_params['batch_size'])

best_val_score = -np.inf
patience_counter = 0
patience = 20

for epoch in range(1, 101):
    model.train()
    for batch in final_train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for batch in final_val_loader:
            batch = batch.to(device)
            out = model(batch)
            val_preds.append(out.cpu().numpy())
            val_trues.append(batch.y.view(-1).cpu().numpy())

    val_preds, val_trues = np.concatenate(val_preds), np.concatenate(val_trues)
    rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
    corr = pearsonr(val_trues, val_preds)[0]
    
    y_range = val_trues.max() - val_trues.min()
    normalized_rmse = rmse / y_range if y_range > 0 else 1.0
    current_val_score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

    print(f"Epoch {epoch:03d} | Validation Score: {current_val_score:.4f} | RMSE: {rmse:.4f} | Corr: {corr:.4f}")

    if current_val_score > best_val_score:
        best_val_score = current_val_score
        patience_counter = 0
        torch.save(model.state_dict(), f'./_save/gnn_rf_top30_best_model.pt')
        print(f"새로운 최고 점수 달성, 모델 저장 완료: {best_val_score:.4f}")
    else:
        patience_counter += 1
        print(f"Score가 개선되지 않았습니다. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"조기 종료! (Epoch: {epoch})")
            break

# ----------------------------------------------------
#               최종 예측 및 저장
# ----------------------------------------------------

final_model_path = './_save/gnn_rf_top30_best_model.pt'
model.load_state_dict(torch.load(final_model_path))
model.eval()

test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        test_preds.append(out.cpu().numpy())

submit_df['Inhibition'] = np.concatenate(test_preds)
submit_df.to_csv('./_save/gnn_rf_top30_submission.csv', index=False)

print("\n✅ 최종 예측 완료: ./_save/gnn_rf_top30_submission.csv")
print(f"  사용된 MACCS 키 개수: {len(top_30_maccs_indices)}")
print(f"  최종 global_feat 차원: {global_feat_dim_updated}")
print(f"  최종 엣지 피처 차원: {final_edge_feature_dim} (선택: {final_edge_feature_choice})")