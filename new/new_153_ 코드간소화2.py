import numpy as np
import pandas as pd
import os, platform
import pickle

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem # Morgan Fingerprint
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import rdmolops # GetAdjacencyMatrix 때문에 추가
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool # GATv2Conv 대신 GINConv 임포트
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import torch.nn.functional as F

import optuna # Optuna 라이브러리 추가

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class Config:
    seed = 42
    epochs = 200
    patience = 20
    lr_patience = 10
    batch_size = 64
    k_splits = 5

    node_feat_dim = 9

    maccs_keys_dim = 167
    morgan_dim = 2048
    use_topN_morgan = 200 
    
    rdkit_basic_desc_count = 200 
    
    remove_outliers = True
    outlier_contamination = 0.02

    lasso_alpha = 0.1
    maccs_lasso_topN = None

    # Optuna 튜닝 대상 하이퍼파라미터의 기본값 또는 범위 설정 (Config에서는 기본값만 설정)
    # GINE에서는 num_heads가 필요 없음
    hidden_dim = 128    
    dropout_rate = 0.2  
    lr = 0.001          
    num_gnn_layers = 3

    morgan_cache_path = './_save/morgan_fps_cache.npy'
    maccs_cache_path = './_save/maccs_keys_cache.npy'
    rdkit_desc_cache_path = './_save/rdkit_descriptors_cache.npy'
    plots_dir = './_save/plots/'

    # Optuna 설정
    optuna_n_trials = 50 # 튜닝 시도 횟수
    optuna_timeout = 3600 # 초 단위 타임아웃 (1시간)
    optuna_direction = 'maximize' # 점수 최대화

def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)
os.makedirs(Config.plots_dir, exist_ok=True)

smart_patterns_raw = ['[C]=[O]', '[NH2]', 'c1ccccc1']
smarts_patterns = [Chem.MolFromSmarts(p) for p in smart_patterns_raw if Chem.MolFromSmarts(p)]
electronegativity = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66, 14: 1.90}
covalent_radius = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39, 14: 1.11}

def atom_features(atom):
    num = atom.GetAtomicNum()
    eneg = electronegativity.get(num, 0)
    radius = covalent_radius.get(num, 0)
    smarts_match = sum(atom.GetOwningMol().HasSubstructMatch(p) for p in smarts_patterns)
    
    return np.array([
        num, 
        eneg, 
        radius,
        atom.GetTotalNumHs(), 
        int(atom.GetHybridization()), 
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()), 
        int(atom.IsInRing()), 
        smarts_match
    ], dtype=np.float32)

def get_morgan_fingerprint(mol, dim=Config.morgan_dim):
    if mol is None:
        return np.zeros(dim, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim), dtype=np.float32)

def get_maccs_keys(mol):
    if mol is None:
        return np.zeros(Config.maccs_keys_dim, dtype=np.float32)
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def get_rdkit_descriptors(mol):
    if mol is None:
        return np.array([np.nan] * len(Descriptors._descList))
    desc_values = []
    for desc_name, desc_func in Descriptors._descList:
        try:
            desc_values.append(desc_func(mol))
        except:
            desc_values.append(np.nan)
    return np.array(desc_values, dtype=np.float32)

def mol_to_graph_data(mol, global_feat, y=None):
    if mol is None:
        return None

    x = []
    edge_index = []

    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(np.array(x), dtype=torch.float)

    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = (torch.tensor(adj) > 0).nonzero(as_tuple=False).t().contiguous()

    if len(edge_index) == 0: # 그래프에 엣지가 없는 경우 (단일 원자 또는 연결되지 않은 그래프)
        if x.size(0) > 0: # 노드는 있는 경우 자기 자신으로 연결 (GINConv 호환을 위해)
            edge_index = torch.tensor([[i, i] for i in range(x.size(0))], dtype=torch.long).t().contiguous()
        else: # 노드도 없는 경우
            return None

    global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).view(1, -1)

    if y is not None:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, y=torch.tensor(y, dtype=torch.float).view(-1))
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor)

# GINE 모델 클래스 정의
class GINWithGlobal(Module): # 클래스 이름 변경
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim, out_dim, num_gnn_layers, dropout_rate): # num_heads 제거
        super(GINWithGlobal, self).__init__()
        self.dropout_rate = dropout_rate

        self.gnn_layers = ModuleList()
        self.bn_layers = ModuleList()
        self.gnn_dropouts = ModuleList()

        # 첫 번째 GINConv 레이어
        # GINConv는 내부적으로 MLP를 가집니다.
        # GINConv(nn.Module, eps=0.0, train_eps=False, **kwargs)
        # 여기서 eps=0.0, train_eps=True는 GINE의 핵심적인 부분으로,
        # 이웃 정보에 대한 가중치(epsilon)를 학습하게 합니다.
        self.gnn_layers.append(GINConv(Linear(node_feat_dim, hidden_dim), eps=0.0, train_eps=True))
        self.bn_layers.append(BatchNorm1d(hidden_dim))
        self.gnn_dropouts.append(Dropout(dropout_rate))

        # 나머지 GINConv 레이어
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GINConv(Linear(hidden_dim, hidden_dim), eps=0.0, train_eps=True))
            self.bn_layers.append(BatchNorm1d(hidden_dim))
            self.gnn_dropouts.append(Dropout(dropout_rate))

        self.pooled_dim = hidden_dim # GIN은 head가 없으므로 hidden_dim 그대로 사용
        self.combined_dim = self.pooled_dim + global_feat_dim

        self.mlp = torch.nn.Sequential(
            Linear(self.combined_dim, hidden_dim * 2),
            BatchNorm1d(hidden_dim * 2),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim * 2, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch, global_feat = data.x, data.edge_index, data.batch, data.global_feat

        for i, conv_layer in enumerate(self.gnn_layers):
            x = conv_layer(x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x) # GINConv는 활성화 함수를 내부에 포함하지 않으므로 ReLU를 명시적으로 적용
            x = self.gnn_dropouts[i](x)

        x = global_mean_pool(x, batch)

        combined_features = torch.cat([x, global_feat.squeeze(1)], dim=1)
        
        return self.mlp(combined_features).view(-1)

def calculate_competition_score(y_true, y_pred):
    pcc = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    A = rmse / 100.0 
    score = 0.5 * (1 - min(A, 1)) + 0.5 * pcc
    
    return score, rmse, pcc

def plot_learning_curves(train_losses, val_rmses, val_pccs, val_scores, fold, current_seed, plots_dir):
    epochs_range = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fold {fold+1} Learning Curves (Seed: {current_seed})', fontsize=16)

    axs[0, 0].plot(epochs_range, train_losses, label='Train Loss')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs_range, val_rmses, label='Val RMSE', color='orange')
    axs[0, 1].set_title('Validation RMSE')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs_range, val_pccs, label='Val PCC', color='green')
    axs[1, 0].set_title('Validation PCC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('PCC')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs_range, val_scores, label='Val Score', color='red')
    axs[1, 1].set_title('Validation Score')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{plots_dir}learning_curve_fold_{fold+1}_seed_{current_seed}.png')
    plt.close(fig)


# Optuna Objective 함수
def objective(trial, train_df_filtered, X_train_global_filtered, y_train_filtered, global_features_combined, train_val_split_indices):
    set_seed(Config.seed) # 각 trial마다 시드 고정 (재현성 위해)

    # Optuna가 탐색할 하이퍼파라미터 정의
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_gnn_layers = trial.suggest_int('num_gnn_layers', 2, 4) # 2, 3, 4 계층

    # Optuna 튜닝 시에는 K-Fold 전체를 돌리지 않고, 첫 번째 Fold 또는 무작위 Fold 하나만 사용
    # 여기서는 미리 전달받은 train_val_split_indices (첫 번째 Fold의 인덱스)를 사용
    train_idx, val_idx = train_val_split_indices

    fold_train_df = train_df_filtered.iloc[train_idx]
    fold_val_df = train_df_filtered.iloc[val_idx]
    
    fold_X_train_global = X_train_global_filtered[train_idx]
    fold_y_train = y_train_filtered[train_idx]
    fold_X_val_global = X_train_global_filtered[val_idx]
    fold_y_val = y_train_filtered[val_idx]

    # 그래프 데이터셋 생성 (tqdm 제거하여 로그 간소화)
    train_data_list = []
    for i in range(len(fold_train_df)):
        smiles = fold_train_df.iloc[i]['Canonical_Smiles']
        mol = Chem.MolFromSmiles(smiles)
        data = mol_to_graph_data(mol, fold_X_train_global[i], fold_y_train[i]) 
        if data is not None:
            train_data_list.append(data)
    
    val_data_list = []
    for i in range(len(fold_val_df)):
        smiles = fold_val_df.iloc[i]['Canonical_Smiles']
        mol = Chem.MolFromSmiles(smiles)
        data = mol_to_graph_data(mol, fold_X_val_global[i], fold_y_val[i])
        if data is not None:
            val_data_list.append(data)

    train_loader = DataLoader(train_data_list, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=Config.batch_size, shuffle=False)

    model = GINWithGlobal( # GINE 모델 사용
        node_feat_dim=Config.node_feat_dim,
        global_feat_dim=global_features_combined.shape[1],
        hidden_dim=hidden_dim, # Optuna에서 제안된 값 사용
        out_dim=1,
        num_gnn_layers=num_gnn_layers, # Optuna에서 제안된 값 사용
        dropout_rate=dropout_rate # Optuna에서 제안된 값 사용
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Optuna에서 제안된 값 사용
    criterion = torch.nn.SmoothL1Loss() 
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=Config.lr_patience, verbose=False) # Optuna 시에는 verbose 끄기

    best_val_score = -np.inf 
    epochs_no_improve = 0
    
    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        model.eval()
        val_preds_original = []
        val_targets_original = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                val_preds_original.extend(out.cpu().numpy())
                val_targets_original.extend(data.y.cpu().numpy())
        
        val_preds_original_clipped = np.clip(val_preds_original, 0, 100)
        val_targets_original_clipped = np.clip(val_targets_original, 0, 100) 

        current_val_score, _, _ = calculate_competition_score(val_targets_original_clipped, val_preds_original_clipped)

        scheduler.step(current_val_score)

        if current_val_score > best_val_score:
            best_val_score = current_val_score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == Config.patience:
                break
        
        # Optuna Pruning: 특정 에포크 이후 성능이 충분히 좋지 않으면 현재 Trial 중단
        trial.report(current_val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_score


if __name__ == '__main__':
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

    combined_smiles_raw = train_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist()
                            
    print(f"총 {len(combined_smiles_raw)}개의 SMILES를 처리합니다.")

    print("\n특성 처리 중 (캐싱 활용)...")

    calculated_morgan_fps = []
    calculated_maccs_keys = []
    calculated_rdkit_descriptors = []

    fp_desc_configs = [
        {"name": "Morgan FP", "path": Config.morgan_cache_path, "gen_func": get_morgan_fingerprint, "target_list": calculated_morgan_fps, "dim_param": Config.morgan_dim},
        {"name": "MACCS Keys", "path": Config.maccs_cache_path, "gen_func": get_maccs_keys, "target_list": calculated_maccs_keys, "dim_param": None},
        {"name": "RDKit Basic Descriptors", "path": Config.rdkit_desc_cache_path, "gen_func": get_rdkit_descriptors, "target_list": calculated_rdkit_descriptors, "dim_param": None},
    ]

    for fp_desc_info in fp_desc_configs:
        name = fp_desc_info["name"]
        path = fp_desc_info["path"]
        gen_func = fp_desc_info["gen_func"]
        target_list = fp_desc_info["target_list"]
        dim_param = fp_desc_info["dim_param"]

        recalculate_fp_desc = False

        if os.path.exists(path):
            try:
                loaded_data = np.load(path)
                if loaded_data.shape[0] != len(combined_smiles_raw):
                    print(f"경고: 로드된 {name} 캐시 파일의 데이터 수가 일치하지 않아 다시 계산합니다.")
                    recalculate_fp_desc = True
                else:
                    target_list.extend(list(loaded_data))
                    print(f"로드된 {name} 데이터 차원: {loaded_data.shape}")
            except Exception as e:
                print(f"경고: {name} 캐시 파일 로드 중 오류 발생 ({e}). 다시 계산합니다.")
                recalculate_fp_desc = True
        else:
            print(f"{name} 캐시 파일이 존재하지 않아 새로 계산합니다.")
            recalculate_fp_desc = True

        if recalculate_fp_desc:
            print(f"{name} 계산 중 (전체 데이터)...")
            temp_fp_list = []
            for i, smiles in enumerate(combined_smiles_raw): # tqdm 제거
                mol = Chem.MolFromSmiles(smiles)
                if dim_param is not None:
                    temp_fp_list.append(gen_func(mol, dim=dim_param))
                else:
                    temp_fp_list.append(gen_func(mol))

            arr = np.array(temp_fp_list)
            target_list.extend(list(arr))
            np.save(path, arr)
            print(f"{name} 캐시 파일 저장 완료. 차원: {target_list[0].shape if target_list else 'N/A'}")
        
    all_morgan_fps_raw = np.array(calculated_morgan_fps)
    all_maccs_keys_raw = np.array(calculated_maccs_keys)
    all_rdkit_descriptors_raw = np.array(calculated_rdkit_descriptors)

    print("모든 특성 계산 또는 로드 완료.")

    train_df = train_df_original.copy()
    test_df = test_df_original.copy()
    
    current_all_morgan_fps = all_morgan_fps_raw.copy()

    morgan_fps_processed = None
    if current_all_morgan_fps.shape[1] > 0:
        lasso_model_morgan = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_morgan.fit(current_all_morgan_fps[:len(train_df)], train_df['Inhibition'].values)
        selector_morgan = SelectFromModel(lasso_model_morgan, max_features=Config.use_topN_morgan, prefit=True, threshold=-np.inf)
        morgan_fps_processed = selector_morgan.transform(current_all_morgan_fps)
    else:
        morgan_fps_processed = np.empty((current_all_morgan_fps.shape[0], 0))
    print(f"Morgan FP 처리된 차원: {morgan_fps_processed.shape}")

    maccs_keys_processed = None
    if all_maccs_keys_raw.shape[1] > 0:
        vt_maccs = VarianceThreshold(threshold=0.0)
        maccs_vt_filtered = vt_maccs.fit_transform(all_maccs_keys_raw)
        print(f"MACCS Keys (VarianceThreshold 필터링 후) 차원: {maccs_vt_filtered.shape}")

        if maccs_vt_filtered.shape[1] > 0:
            lasso_model_maccs = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
            lasso_model_maccs.fit(maccs_vt_filtered[:len(train_df_original)], train_df_original['Inhibition'].values)
            
            selector_maccs = SelectFromModel(lasso_model_maccs, max_features=Config.maccs_lasso_topN, prefit=True, threshold=-np.inf)
            maccs_keys_processed = selector_maccs.transform(maccs_vt_filtered)
        else:
            maccs_keys_processed = np.empty((all_maccs_keys_raw.shape[0], 0))
    else:
        maccs_keys_processed = np.empty((all_maccs_keys_raw.shape[0], 0))
    print(f"MACCS Keys (최종 처리 후) 차원: {maccs_keys_processed.shape}")

    rdkit_desc_processed = None
    if all_rdkit_descriptors_raw.shape[1] > 0:
        temp_rdkit_df = pd.DataFrame(all_rdkit_descriptors_raw)
        temp_rdkit_df = temp_rdkit_df.fillna(temp_rdkit_df.mean())
        
        vt_rdkit = VarianceThreshold(threshold=0.0)
        rdkit_vt_filtered = vt_rdkit.fit_transform(temp_rdkit_df.values)
        print(f"RDKit Descriptors (VarianceThreshold 필터링 후) 차원: {rdkit_vt_filtered.shape}")

        if rdkit_vt_filtered.shape[1] > 0:
            scaler_rdkit = StandardScaler()
            rdkit_scaled = scaler_rdkit.fit_transform(rdkit_vt_filtered)
            
            lasso_model_rdkit = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
            lasso_model_rdkit.fit(rdkit_scaled[:len(train_df_original)], train_df_original['Inhibition'].values)
            selector_rdkit = SelectFromModel(lasso_model_rdkit, max_features=Config.rdkit_basic_desc_count, prefit=True, threshold=-np.inf)
            rdkit_desc_processed = selector_rdkit.transform(rdkit_scaled)
        else:
            rdkit_desc_processed = np.empty((all_rdkit_descriptors_raw.shape[0], 0))
    else:
        rdkit_desc_processed = np.empty((all_rdkit_descriptors_raw.shape[0], 0))
    print(f"RDKit Descriptors (최종 처리 후) 차원: {rdkit_desc_processed.shape}")


    global_features_combined = np.hstack([
        morgan_fps_processed,
        maccs_keys_processed,
        rdkit_desc_processed,
    ])
    print(f"모든 Global Features 결합된 차원: {global_features_combined.shape}")

    X_train_global = global_features_combined[:len(train_df_original)]
    y_train_original = train_df_original['Inhibition'].values

    if Config.remove_outliers:
        print("훈련 데이터에서 아웃라이어 제거 중...")
        iso_forest = IsolationForest(contamination=Config.outlier_contamination, random_state=Config.seed, n_jobs=-1)
        outlier_preds = iso_forest.fit_predict(X_train_global)
        
        train_df_filtered = train_df_original[outlier_preds == 1].reset_index(drop=True)
        X_train_global_filtered = X_train_global[outlier_preds == 1]
        y_train_filtered = y_train_original[outlier_preds == 1]
        print(f"아웃라이어 제거 후 훈련 데이터 수: {len(train_df_filtered)} (제거된 수: {len(train_df_original) - len(train_df_filtered)})")
    else:
        train_df_filtered = train_df_original
        X_train_global_filtered = X_train_global
        y_train_filtered = y_train_original
    
    X_test_global = global_features_combined[len(train_df_original):]
    
    current_seed = Config.seed
    set_seed(current_seed)

    if 'Inhibition_bins' not in train_df_filtered.columns:
        train_df_filtered['Inhibition_bins'] = pd.cut(
            train_df_filtered['Inhibition'],
            bins=10,
            labels=False,
            include_lowest=True
        )
    
    kf = StratifiedKFold(n_splits=Config.k_splits, shuffle=True, random_state=current_seed)
    
    # Optuna 튜닝을 위한 단일 Fold의 train_idx, val_idx만 추출
    # Optuna는 전체 K-Fold를 돌리지 않고, 빠르게 하이퍼파라미터를 탐색합니다.
    first_fold_train_idx, first_fold_val_idx = next(iter(kf.split(train_df_filtered, train_df_filtered['Inhibition_bins'])))
    train_val_split_indices = (first_fold_train_idx, first_fold_val_idx)


    print("\n--- Optuna 하이퍼파라미터 튜닝 시작 ---")
    # Optuna Study 생성
    # sampler는 TPE (Tree-structured Parzen Estimator) 사용이 일반적
    # pruner는 ASHA pruner를 사용하여 성능이 낮은 trial을 조기에 중단
    study = optuna.create_study(
        direction=Config.optuna_direction,
        sampler=optuna.samplers.TPESampler(seed=Config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    )

    # objective 함수 실행
    study.optimize(
        lambda trial: objective(trial, train_df_filtered, X_train_global_filtered, y_train_filtered, global_features_combined, train_val_split_indices),
        n_trials=Config.optuna_n_trials,
        timeout=Config.optuna_timeout,
        show_progress_bar=False # 진행바 표시를 위해 tqdm 대신 False로 설정
    )

    print("\n--- Optuna 튜닝 완료 ---")
    print(f"최적의 하이퍼파라미터: {study.best_params}")
    print(f"최고 점수 (Best Trial Score): {study.best_value:.4f}")

    # Optuna가 찾은 최적의 하이퍼파라미터로 Config 업데이트
    Config.hidden_dim = study.best_params['hidden_dim']
    Config.dropout_rate = study.best_params['dropout_rate']
    Config.lr = study.best_params['lr']
    Config.num_gnn_layers = study.best_params['num_gnn_layers']

    print("\n--- 최적화된 하이퍼파라미터로 K-Fold 교차 검증 및 모델 학습 시작 ---")

    test_predictions_folds = []
    
    fold_val_scores_for_current_seed = []
    fold_val_rmses_for_current_seed = []
    fold_val_pccs_for_current_seed = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_filtered, train_df_filtered['Inhibition_bins'])):
        print(f"\n--- Fold {fold+1}/{Config.k_splits} ---")
        
        fold_train_df = train_df_filtered.iloc[train_idx]
        fold_val_df = train_df_filtered.iloc[val_idx]
        
        fold_X_train_global = X_train_global_filtered[train_idx]
        fold_y_train = fold_train_df['Inhibition'].values
        fold_X_val_global = X_train_global_filtered[val_idx]
        fold_y_val = fold_val_df['Inhibition'].values

        train_data_list = []
        for i in range(len(fold_train_df)): # tqdm 제거
            smiles = fold_train_df.iloc[i]['Canonical_Smiles']
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data(mol, fold_X_train_global[i], fold_y_train[i]) 
            if data is not None:
                train_data_list.append(data)
        
        val_data_list = []
        for i in range(len(fold_val_df)): # tqdm 제거
            smiles = fold_val_df.iloc[i]['Canonical_Smiles']
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data(mol, fold_X_val_global[i], fold_y_val[i])
            if data is not None:
                val_data_list.append(data)

        train_loader = DataLoader(train_data_list, batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=Config.batch_size, shuffle=False)

        model = GINWithGlobal( # GINE 모델 사용
            node_feat_dim=Config.node_feat_dim,
            global_feat_dim=global_features_combined.shape[1],
            hidden_dim=Config.hidden_dim, # Optuna에서 찾은 최적의 값 사용
            out_dim=1,
            num_gnn_layers=Config.num_gnn_layers, # Optuna에서 찾은 최적의 값 사용
            dropout_rate=Config.dropout_rate # Optuna에서 찾은 최적의 값 사용
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr) # Optuna에서 찾은 최적의 값 사용
        criterion = torch.nn.SmoothL1Loss() 
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=Config.lr_patience, verbose=True)

        best_val_score = -np.inf 
        best_val_rmse_at_best_score = float('inf') 
        best_val_pcc_at_best_score = -np.inf 
        epochs_no_improve = 0

        train_losses = []
        val_rmses = []
        val_pccs = []
        val_scores = []
        
        for epoch in range(1, Config.epochs + 1):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y) 
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            val_preds_original = []
            val_targets_original = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    val_preds_original.extend(out.cpu().numpy())
                    val_targets_original.extend(data.y.cpu().numpy())
            
            val_preds_original_clipped = np.clip(val_preds_original, 0, 100)
            val_targets_original_clipped = np.clip(val_targets_original, 0, 100) 

            current_val_score, val_rmse, val_pcc = calculate_competition_score(val_targets_original_clipped, val_preds_original_clipped)

            scheduler.step(current_val_score)

            train_losses.append(avg_train_loss)
            val_rmses.append(val_rmse)
            val_pccs.append(val_pcc)
            val_scores.append(current_val_score)

            if current_val_score > best_val_score:
                best_val_score = current_val_score
                best_val_rmse_at_best_score = val_rmse
                best_val_pcc_at_best_score = val_pcc
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'./_save/best_model_fold_{fold}_seed{current_seed}_GIN_H{Config.hidden_dim}_DP{Config.dropout_rate}_LR{Config.lr}_GNNL{Config.num_gnn_layers}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == Config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == 1 or epoch == len(train_losses):
                print(f"Epoch {epoch}/{Config.epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, Val Score: {current_val_score:.4f}")

        print(f"\n--- Fold {fold+1} Summary (Seed: {current_seed}) ---")
        print(f"Best Val Score: {best_val_score:.4f} (Corresponding RMSE: {best_val_rmse_at_best_score:.4f}, PCC: {best_val_pcc_at_best_score:.4f})")
        plot_learning_curves(train_losses, val_rmses, val_pccs, val_scores, fold, current_seed, Config.plots_dir)
        
        fold_val_scores_for_current_seed.append(best_val_score)
        fold_val_rmses_for_current_seed.append(best_val_rmse_at_best_score)
        fold_val_pccs_for_current_seed.append(best_val_pcc_at_best_score)

        model.load_state_dict(torch.load(
            f'./_save/best_model_fold_{fold}_seed{current_seed}_GIN_H{Config.hidden_dim}_DP{Config.dropout_rate}_LR{Config.lr}_GNNL{Config.num_gnn_layers}.pth'))
        model.eval()

        test_data_list = []
        for i in range(len(test_df_original)): # tqdm 제거
            smiles = test_df_original.iloc[i]['Canonical_Smiles']
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data(mol, X_test_global[i])
            if data is not None:
                test_data_list.append(data)
        test_loader = DataLoader(test_data_list, batch_size=Config.batch_size, shuffle=False)

        fold_test_preds_original = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                fold_test_preds_original.extend(out.cpu().numpy())
        
        fold_test_preds_clipped = np.clip(fold_test_preds_original, 0, 100)
        
        test_predictions_folds.append(fold_test_preds_clipped)

    final_preds_mean = np.mean(test_predictions_folds, axis=0)

    avg_score_current_seed = np.mean(fold_val_scores_for_current_seed)
    avg_rmse_current_seed = np.mean(fold_val_rmses_for_current_seed)
    avg_pcc_current_seed = np.mean(fold_val_pccs_for_current_seed)
    print(f"\n--- 최종 (단일 시드: {current_seed}) 평균 Val Metrics ---")
    print(f"Average Val Score: {avg_score_current_seed:.4f}")
    print(f"Average Val RMSE: {avg_rmse_current_seed:.4f}")
    print(f"Average Val PCC: {avg_pcc_current_seed:.4f}")
    print("-" * 50)


    print("\n--- 최종 예측값 계산 완료 ---")
    final_preds_clipped = np.clip(final_preds_mean, 0, 100)

    submit_df = submit_df_original.copy()
    submit_df['Inhibition'] = final_preds_clipped
    
    submission_filename = (f"submission_GIN_Optuna_H{Config.hidden_dim}"
                           f"_DP{Config.dropout_rate}_LR{Config.lr}_GNNL{Config.num_gnn_layers}_RDKit_BestNodeFeat_SmoothL1Loss.csv") # 파일명 업데이트
    submit_df.to_csv(f'./_save/{submission_filename}', index=False, encoding='utf-8')
    print(f"\n최종 Submission 파일 '{f'./_save/{submission_filename}'}'이(가) UTF-8로 성공적으로 생성되었습니다.")