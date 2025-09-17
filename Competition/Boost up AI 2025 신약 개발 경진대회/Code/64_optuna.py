import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
import pickle # 캐시 저장/로드를 위해 pickle 사용

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import RDKFingerprint
from mordred import Calculator, descriptors
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class Config:
    seed = 42
    epochs = 200
    patience = 20
    lr_patience = 10
    batch_size = 64
    lr = 0.001
    k_splits = 5

    node_feat_dim = 9 # 원자 피처 (고정)

    maccs_keys_dim = 167 # MACCS Keys 차원 (고정)
    morgan_dim = 2048 # Morgan Fingerprint 원본 차원
    use_topN_morgan = 200 # Morgan FP에서 선택할 Top N 개수
    mordred_dim = 60 # Mordred 디스크립터에서 선택할 Top N 개수

    apfp_target_dim = 2048 # APFP를 패딩할 목표 차원
    tfp_initial_dim = 2048 # Topological Fingerprint 원본 차원

    rdkit_basic_desc_count = 5 # RDKit 기본 디스크립터 개수

    remove_outliers = True
    outlier_contamination = 0.02

    lasso_alpha = 0.1 # Lasso alpha 값

    tuning_apfp_n = 200 # 이전 최적값으로 고정
    tuning_tfp_n = 200  # 이전 최적값으로 고정

    # ✨✨✨ 새롭게 추가된 GNN 하이퍼파라미터 튜닝 값들 ✨✨✨
    tuning_hidden_dims = [64, 128, 256] # GATv2 hidden_dim 후보군
    tuning_num_heads = [4, 6, 8]       # GATv2 num_heads 후보군
    tuning_dropout_rates = [0.3, 0.4, 0.5] # Dropout rate 후보군
    tuning_lrs = [0.001, 0.0005] # Learning rate 후보군

    mordred_cache_path = './_save/mordred_combined_cache.csv'
    morgan_cache_path = './_save/morgan_fps_cache.npy'
    maccs_cache_path = './_save/maccs_keys_cache.npy'
    apfp_cache_path = './_save/apfp_cache.npy'
    tfp_cache_path = './_save/tfp_cache.npy'
    rdkit_desc_cache_path = './_save/rdkit_desc_cache.npy'

# 시드 고정
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./_save', exist_ok=True)

def atom_features(atom):
    return np.array([
        atom.GetSymbol() == 'C', atom.GetSymbol() == 'N', atom.GetSymbol() == 'O',
        atom.GetSymbol() == 'S', atom.GetSymbol() == 'F', atom.GetSymbol() == 'Cl',
        atom.GetSymbol() == 'Br', atom.GetSymbol() == 'I', atom.GetDegree(),
    ], dtype=np.float32)

calc = Calculator(descriptors, ignore_3D=True)

def get_morgan_fingerprint(mol, dim=Config.morgan_dim):
    if mol is None:
        return np.zeros(dim, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim), dtype=np.float32)

def get_maccs_keys(mol):
    if mol is None:
        return np.zeros(Config.maccs_keys_dim, dtype=np.float32)
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def get_atom_pair_fingerprint(mol):
    if mol is None:
        return np.array([], dtype=np.float32) # Return empty array if mol is None
    return np.array(Pairs.GetAtomPairFingerprintAsBitVect(mol), dtype=np.float32)

def get_topological_fingerprint(mol, dim=Config.tfp_initial_dim):
    if mol is None:
        return np.zeros(dim, dtype=np.float32)
    return np.array(RDKFingerprint(mol, fpSize=dim), dtype=np.float32)

def get_rdkit_descriptors(mol):
    if mol is None:
        return np.array([0.0] * Config.rdkit_basic_desc_count, dtype=np.float32)

    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return np.array([logp, tpsa, mw, hbd, hba], dtype=np.float32)

def standardize_fingerprints(fp_list, target_dim, dtype=np.float32):
    if not fp_list:
        return np.empty((0, target_dim), dtype=dtype)
    standardized_fps = []
    for fp in fp_list:
        # ⭐️ 변경된 부분: 빈 배열도 처리 가능하도록 함
        if fp.size == 0:
            standardized_fps.append(np.zeros(target_dim, dtype=dtype))
        elif fp.shape[0] < target_dim:
            padded_fp = np.pad(fp, (0, target_dim - fp.shape[0]), 'constant')
            standardized_fps.append(padded_fp)
        else:
            standardized_fps.append(fp[:target_dim])
    return np.array(standardized_fps, dtype=dtype)

def mol_to_graph_data(mol, global_feat, y=None):
    if mol is None: # ⭐️ 추가된 부분: mol이 None이면 처리하지 않고 None 반환
        return None

    x = []
    edge_index = []

    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(np.array(x), dtype=torch.float)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])

    if len(edge_index) == 0:
        # 단일 원자 또는 연결이 없는 경우를 위한 처리
        if x.size(0) > 0: # 원자가 하나라도 있으면 자기 루프 추가
            edge_index = torch.tensor([[0,0]], dtype=torch.long)
        else: # 원자가 전혀 없는 경우 (거의 발생하지 않지만 안전을 위해)
            return None # 그래프를 만들 수 없으므로 None 반환
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).view(1, -1)

    if y is not None:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, y=torch.tensor(y, dtype=torch.float).view(-1))
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor)

class GATv2WithGlobal(Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim, out_dim, num_heads, dropout_rate):
        super(GATv2WithGlobal, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim

        self.conv1 = GATv2Conv(node_feat_dim, hidden_dim, heads=num_heads, dropout=dropout_rate, add_self_loops=True)
        self.bn1 = BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, add_self_loops=True)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, add_self_loops=True)
        self.bn3 = BatchNorm1d(hidden_dim * num_heads)

        self.pooled_dim = hidden_dim * num_heads
        self.combined_dim = self.pooled_dim + self.global_feat_dim

        self.fc1 = Linear(self.combined_dim, hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch, global_feat = data.x, data.edge_index, data.batch, data.global_feat

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        combined_features = torch.cat([x, global_feat.squeeze(1)], dim=1)

        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(-1)

if __name__ == '__main__':
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

    # ⭐️ 수정된 부분: combined_smiles_raw에서 중복 제거
    combined_smiles_raw = train_df_original['Canonical_Smiles'].tolist() + \
                            test_df_original['Canonical_Smiles'].tolist()
                            
    print(f"총 {len(combined_smiles_raw)}개의 SMILES를 처리합니다.")

    print("Mordred Descriptors 처리 중 (캐싱 활용)...")
    mordred_combined_df = None
    recalculate_mordred = False

    if os.path.exists(Config.mordred_cache_path):
        try:
            mordred_combined_df = pd.read_csv(Config.mordred_cache_path)
            if mordred_combined_df.empty or mordred_combined_df.shape[0] != len(combined_smiles_raw):
                print("경고: 로드된 Mordred 캐시 파일이 비어있거나 유효하지 않거나 데이터 수가 일치하지 않아 다시 계산합니다.")
                recalculate_mordred = True
            else:
                print(f"로드된 Mordred 데이터 차원: {mordred_combined_df.shape}")
        except pd.errors.EmptyDataError:
            print("경고: Mordred 캐시 파일이 비어있어 다시 계산합니다.")
            recalculate_mordred = True
        except Exception as e:
            print(f"경고: Mordred 캐시 파일 로드 중 오류 발생 ({e}). 다시 계산합니다.")
            recalculate_mordred = True
    else:
        print("Mordred Descriptors 캐시 파일이 존재하지 않아 새로 계산합니다.")
        recalculate_mordred = True

    if recalculate_mordred:
        print("Mordred Descriptors 계산 중 (전체 데이터)...")
        # ⭐️ 수정된 부분: combined_mols_raw 대신 combined_smiles_raw를 직접 사용하고 각 SMILES마다 mol 생성
        combined_mols_raw = [] # Mordred 계산을 위해 몰 객체 리스트 생성
        for smiles in tqdm(combined_smiles_raw, desc="SMILES -> Mol 변환 (Mordred)"):
            mol = Chem.MolFromSmiles(smiles)
            combined_mols_raw.append(mol)

        try:
            mordred_df_temp = calc.pandas(combined_mols_raw, n_jobs=-1, quiet=True)
            
            mordred_combined_df = mordred_df_temp.select_dtypes(include=np.number).fillna(0)
            
            mordred_combined_df = mordred_combined_df.loc[:, mordred_combined_df.apply(pd.Series.nunique) != 1]

            print(f"\nMordred 계산 성공. 총 디스크립터 차원: {mordred_combined_df.shape[1]}")
        except Exception as e:
            print(f"ERROR: Mordred 계산 중 오류 발생: {type(e).__name__} - {e}")
            print("Mordred 계산에 실패하여 빈 데이터프레임으로 대체합니다.")
            mordred_combined_df = pd.DataFrame(np.zeros((len(combined_smiles_raw), 0)), columns=[])
        
        mordred_combined_df.to_csv(Config.mordred_cache_path, index=False)
        print("Mordred Descriptors 캐시 파일 저장 완료.")

    print("\n모든 Fingerprint 및 RDKit Descriptors 처리 중 (캐싱 활용)...")

    calculated_morgan_fps = []
    calculated_maccs_keys = []
    calculated_apfp = []
    calculated_tfp = []
    calculated_rdkit_descriptors = []

    fp_desc_configs = [
        {"name": "Morgan FP", "path": Config.morgan_cache_path, "gen_func": get_morgan_fingerprint, "target_list": calculated_morgan_fps, "dim_param": Config.morgan_dim},
        {"name": "MACCS Keys", "path": Config.maccs_cache_path, "gen_func": get_maccs_keys, "target_list": calculated_maccs_keys, "dim_param": None},
        {"name": "APFP", "path": Config.apfp_cache_path, "gen_func": get_atom_pair_fingerprint, "target_list": calculated_apfp, "dim_param": None},
        {"name": "TFP", "path": Config.tfp_cache_path, "gen_func": get_topological_fingerprint, "target_list": calculated_tfp, "dim_param": Config.tfp_initial_dim},
        {"name": "RDKit Descriptors", "path": Config.rdkit_desc_cache_path, "gen_func": get_rdkit_descriptors, "target_list": calculated_rdkit_descriptors, "dim_param": None}
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
            for i, smiles in tqdm(enumerate(combined_smiles_raw), total=len(combined_smiles_raw), desc=f"Calculating {name}"):
                mol = Chem.MolFromSmiles(smiles) # ⭐️ 수정된 부분: 여기서 mol 객체 생성
                if dim_param is not None:
                    temp_fp_list.append(gen_func(mol, dim=dim_param))
                else:
                    temp_fp_list.append(gen_func(mol)) # ⭐️ 수정된 부분: get_*_fingerprint 함수들이 None을 처리하도록 함

            if name == "APFP":
                processed_fps = standardize_fingerprints(temp_fp_list, Config.apfp_target_dim)
                target_list.extend(list(processed_fps))
                np.save(path, processed_fps)
            else:
                arr = np.array(temp_fp_list)
                target_list.extend(list(arr))
                np.save(path, arr)
            print(f"{name} 캐시 파일 저장 완료. 차원: {target_list[0].shape if target_list else 'N/A'}")
        
    all_morgan_fps_raw = np.array(calculated_morgan_fps)
    all_maccs_keys_raw = np.array(calculated_maccs_keys)
    all_apfp_raw = np.array(calculated_apfp)
    all_tfp_raw = np.array(calculated_tfp)
    all_rdkit_descriptors_raw = np.array(calculated_rdkit_descriptors)

    print("모든 Fingerprint 및 RDKit Descriptors 계산 또는 로드 완료.")

    # ----------------------------------------------------------------------------------------------------
    # ✨✨✨ GNN 하이퍼파라미터 튜닝 루프 시작 ✨✨✨
    # ----------------------------------------------------------------------------------------------------
    gnn_tuning_results = {} # GNN 하이퍼파라미터 튜닝 결과를 저장할 딕셔너리
    best_gnn_overall_score = -float('inf')
    best_gnn_params = {}
    final_best_test_predictions = None

    for current_hidden_dim in Config.tuning_hidden_dims:
        for current_num_heads in Config.tuning_num_heads:
            for current_dropout_rate in Config.tuning_dropout_rates:
                for current_lr in Config.tuning_lrs:
                    print(f"\n===== GNN 튜닝 조합: hidden_dim={current_hidden_dim}, num_heads={current_num_heads}, dropout_rate={current_dropout_rate}, lr={current_lr} =====")
                    
                    # 튜닝 중인 GNN 하이퍼파라미터를 Config에 임시로 업데이트
                    Config.hidden_dim = current_hidden_dim
                    Config.num_heads = current_num_heads
                    Config.dropout_rate = current_dropout_rate
                    Config.lr = current_lr

                    # 피처 전처리 (이 부분은 GNN 하이퍼파라미터 튜닝과 직접적인 관련이 없으므로,
                    # 이전 최적 APFP N=200, TFP N=200을 사용하여 고정)
                    # 만약 APFP/TFP도 함께 튜닝하려면 이 부분을 튜닝 루프 안에 넣어야 합니다.

                    train_df = train_df_original.copy()
                    test_df = test_df_original.copy()
                    
                    current_all_morgan_fps = all_morgan_fps_raw.copy()
                    current_all_apfp = all_apfp_raw.copy()
                    current_all_tfp = all_tfp_raw.copy()
                    current_all_rdkit_descriptors = all_rdkit_descriptors_raw.copy()

                    # Mordred 특성 전처리
                    mordred_processed = None
                    if mordred_combined_df.shape[1] == 0:
                        mordred_processed = np.empty((mordred_combined_df.shape[0], 0))
                    else:
                        vt = VarianceThreshold(threshold=0.0)
                        mordred_vt = vt.fit_transform(mordred_combined_df)
                        mordred_vt_cols = mordred_combined_df.columns[vt.get_support()]

                        scaler = StandardScaler()
                        mordred_scaled = scaler.fit_transform(mordred_vt)

                        corr_threshold = 0.95
                        mordred_df_temp = pd.DataFrame(mordred_scaled, columns=mordred_vt_cols)

                        if mordred_df_temp.shape[1] > 1:
                            corr_matrix = mordred_df_temp.corr().abs()
                            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
                        else:
                            to_drop = []

                        if to_drop:
                            mordred_filtered_corr_df = mordred_df_temp.drop(columns=to_drop)
                            mordred_filtered_corr = mordred_filtered_corr_df.values
                        else:
                            mordred_filtered_corr_df = mordred_df_temp
                            mordred_filtered_corr = mordred_scaled

                        if mordred_filtered_corr.shape[1] > 0:
                            lasso_model_mordred = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                            lasso_model_mordred.fit(mordred_filtered_corr[:len(train_df)], train_df['Inhibition'].values)
                            selector_mordred = SelectFromModel(lasso_model_mordred, max_features=Config.mordred_dim, prefit=True, threshold=-np.inf)
                            mordred_processed = selector_mordred.transform(mordred_filtered_corr)
                        else:
                            mordred_processed = np.empty((mordred_filtered_corr.shape[0], 0))
                    print(f"Mordred 처리된 차원: {mordred_processed.shape}")

                    # Morgan Fingerprint 피처 선택 (Config.use_topN_morgan 사용)
                    morgan_fps_processed = None
                    if current_all_morgan_fps.shape[1] > 0:
                        lasso_model_morgan = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                        lasso_model_morgan.fit(current_all_morgan_fps[:len(train_df)], train_df['Inhibition'].values)
                        selector_morgan = SelectFromModel(lasso_model_morgan, max_features=Config.use_topN_morgan, prefit=True, threshold=-np.inf)
                        morgan_fps_processed = selector_morgan.transform(current_all_morgan_fps)
                    else:
                        morgan_fps_processed = np.empty((current_all_morgan_fps.shape[0], 0))
                    print(f"Morgan FP 처리된 차원: {morgan_fps_processed.shape}")

                    # APFP 피처 선택 (Config.tuning_apfp_n 사용)
                    apfp_processed = None
                    if current_all_apfp.shape[1] > 0:
                        lasso_model_apfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                        lasso_model_apfp.fit(current_all_apfp[:len(train_df)], train_df['Inhibition'].values)
                        selector_apfp = SelectFromModel(lasso_model_apfp, max_features=Config.tuning_apfp_n, prefit=True, threshold=-np.inf)
                        apfp_processed = selector_apfp.transform(current_all_apfp)
                    else:
                        apfp_processed = np.empty((current_all_apfp.shape[0], 0))
                    print(f"APFP 처리된 차원: {apfp_processed.shape}")

                    # TFP 피처 선택 (Config.tuning_tfp_n 사용)
                    tfp_processed = None
                    if current_all_tfp.shape[1] > 0:
                        lasso_model_tfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                        lasso_model_tfp.fit(current_all_tfp[:len(train_df)], train_df['Inhibition'].values)
                        selector_tfp = SelectFromModel(lasso_model_tfp, max_features=Config.tuning_tfp_n, prefit=True, threshold=-np.inf)
                        tfp_processed = selector_tfp.transform(current_all_tfp)
                    else:
                        tfp_processed = np.empty((current_all_tfp.shape[0], 0))
                    print(f"TFP 처리된 차원: {tfp_processed.shape}")

                    # RDKit 기본 디스크립터 전처리
                    rdkit_desc_processed = None
                    if current_all_rdkit_descriptors.shape[1] > 0:
                        scaler_rdkit = StandardScaler()
                        rdkit_scaled = scaler_rdkit.fit_transform(current_all_rdkit_descriptors)

                        lasso_model_rdkit = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                        lasso_model_rdkit.fit(rdkit_scaled[:len(train_df)], train_df['Inhibition'].values)
                        selector_rdkit = SelectFromModel(lasso_model_rdkit, prefit=True, threshold=-np.inf)
                        rdkit_desc_processed = selector_rdkit.transform(rdkit_scaled)
                    else:
                        rdkit_desc_processed = np.empty((current_all_rdkit_descriptors.shape[0], 0))
                    print(f"RDKit 디스크립터 처리된 차원: {rdkit_desc_processed.shape}")

                    # 모든 전처리된 피처 결합
                    global_features_combined = np.hstack([
                        mordred_processed,
                        morgan_fps_processed,
                        all_maccs_keys_raw, # MACCS Keys는 Top N 선택 없이 모두 사용
                        apfp_processed,
                        tfp_processed,
                        rdkit_desc_processed
                    ])
                    print(f"모든 Global Features 결합된 차원: {global_features_combined.shape}")

                    X_train_global = global_features_combined[:len(train_df_original)] # original train_df_original 길이로 분리
                    y_train_original = train_df_original['Inhibition'].values # original train_df_original의 y값

                    # 아웃라이어 제거 로직
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
                    
                    X_test_global = global_features_combined[len(train_df_original):] # test_df_original 길이만큼만 추출
                    
                    kf = KFold(n_splits=Config.k_splits, shuffle=True, random_state=Config.seed)
                    fold_results = []
                    test_preds_folds_current_gnn_params = [] # 현재 GNN 파라미터 조합에 대한 test_preds_folds

                    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_filtered)):
                        print(f"\n--- Fold {fold+1}/{Config.k_splits} ---")
                        
                        fold_train_df = train_df_filtered.iloc[train_idx]
                        fold_val_df = train_df_filtered.iloc[val_idx]
                        
                        fold_X_train_global = X_train_global_filtered[train_idx]
                        fold_y_train = y_train_filtered[train_idx]
                        fold_X_val_global = X_train_global_filtered[val_idx]
                        fold_y_val = y_train_filtered[val_idx]

                        # Graph 데이터 생성
                        train_data_list = []
                        for i in tqdm(range(len(fold_train_df)), desc="Train Graph Data"):
                            smiles = fold_train_df.iloc[i]['Canonical_Smiles']
                            mol = Chem.MolFromSmiles(smiles)
                            data = mol_to_graph_data(mol, fold_X_train_global[i], fold_y_train[i])
                            if data is not None:
                                train_data_list.append(data)
                        
                        val_data_list = []
                        for i in tqdm(range(len(fold_val_df)), desc="Validation Graph Data"):
                            smiles = fold_val_df.iloc[i]['Canonical_Smiles']
                            mol = Chem.MolFromSmiles(smiles)
                            data = mol_to_graph_data(mol, fold_X_val_global[i], fold_y_val[i])
                            if data is not None:
                                val_data_list.append(data)

                        train_loader = DataLoader(train_data_list, batch_size=Config.batch_size, shuffle=True)
                        val_loader = DataLoader(val_data_list, batch_size=Config.batch_size, shuffle=False)

                        # 모델 초기화 (현재 튜닝 중인 GNN 하이퍼파라미터 사용)
                        model = GATv2WithGlobal(
                            node_feat_dim=Config.node_feat_dim,
                            global_feat_dim=global_features_combined.shape[1], # 결합된 글로벌 피처 차원
                            hidden_dim=Config.hidden_dim, # ✨ 튜닝 변수 적용
                            out_dim=1,
                            num_heads=Config.num_heads, # ✨ 튜닝 변수 적용
                            dropout_rate=Config.dropout_rate # ✨ 튜닝 변수 적용
                        ).to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr) # ✨ 튜닝 변수 적용
                        criterion = torch.nn.MSELoss()
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=Config.lr_patience, verbose=True)

                        best_val_rmse = float('inf')
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

                            avg_train_loss = total_loss / len(train_loader.dataset)
                            
                            # Validation
                            model.eval()
                            val_preds = []
                            val_targets = []
                            with torch.no_grad():
                                for data in val_loader:
                                    data = data.to(device)
                                    out = model(data)
                                    val_preds.extend(out.cpu().numpy())
                                    val_targets.extend(data.y.cpu().numpy())
                            
                            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
                            val_pcc = pearsonr(val_targets, val_preds)[0] if len(val_targets) > 1 else 0.0

                            scheduler.step(val_rmse)

                            # Early stopping
                            if val_rmse < best_val_rmse:
                                best_val_rmse = val_rmse
                                epochs_no_improve = 0
                                # Save best model for this fold and current GNN params
                                torch.save(model.state_dict(), f'best_model_fold_{fold}_h{current_hidden_dim}_nh{current_num_heads}_dp{current_dropout_rate}_lr{current_lr}.pth')
                            else:
                                epochs_no_improve += 1
                                if epochs_no_improve == Config.patience:
                                    print(f"Early stopping at epoch {epoch}")
                                    break
                            
                            if epoch % 10 == 0 or epoch == 1 or epoch == Config.epochs:
                                print(f"Epoch {epoch}/{Config.epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}")

                        # Load best model for evaluation and test prediction
                        model.load_state_dict(torch.load(f'best_model_fold_{fold}_h{current_hidden_dim}_nh{current_num_heads}_dp{current_dropout_rate}_lr{current_lr}.pth'))
                        model.eval()

                        final_val_preds = []
                        final_val_targets = []
                        with torch.no_grad():
                            for data in val_loader:
                                data = data.to(device)
                                out = model(data)
                                final_val_preds.extend(out.cpu().numpy())
                                final_val_targets.extend(data.y.cpu().numpy())
                        
                        final_val_rmse = np.sqrt(mean_squared_error(final_val_targets, final_val_preds))
                        final_val_pcc = pearsonr(final_val_targets, final_val_preds)[0] if len(final_val_targets) > 1 else 0.0
                        print(f"Fold {fold+1} Best Val RMSE: {final_val_rmse:.4f}, Best Val PCC: {final_val_pcc:.4f}")
                        fold_results.append({'rmse': final_val_rmse, 'pcc': final_val_pcc})

                        # Test prediction for this fold
                        test_data_list = []
                        # test_df_original을 사용하여 mol_to_graph_data를 호출 (원래 test set)
                        for i in tqdm(range(len(test_df_original)), desc="Test Graph Data"):
                            smiles = test_df_original.iloc[i]['Canonical_Smiles']
                            mol = Chem.MolFromSmiles(smiles)
                            # X_test_global은 이미 test_df_original에 해당하는 부분입니다.
                            data = mol_to_graph_data(mol, X_test_global[i])
                            if data is not None:
                                test_data_list.append(data)
                        
                        test_loader = DataLoader(test_data_list, batch_size=Config.batch_size, shuffle=False)
                        
                        fold_test_preds = []
                        with torch.no_grad():
                            for data in test_loader:
                                data = data.to(device)
                                out = model(data)
                                fold_test_preds.extend(out.cpu().numpy())
                        test_preds_folds_current_gnn_params.append(np.array(fold_test_preds))

                    # ----------------------------------------------------------------------------------------------------
                    # 🔧 GNN 조합별 결과 집계 및 저장
                    # ----------------------------------------------------------------------------------------------------
                    avg_rmse = np.mean([res['rmse'] for res in fold_results])
                    avg_pcc = np.mean([res['pcc'] for res in fold_results])
                    print(f"\nAverage RMSE across folds for current GNN params: {avg_rmse:.4f}")
                    print(f"Average PCC across folds for current GNN params: {avg_pcc:.4f}")
                    
                    gnn_param_key = f"H{current_hidden_dim}_NH{current_num_heads}_DP{current_dropout_rate}_LR{current_lr}"
                    gnn_tuning_results[gnn_param_key] = {'rmse': avg_rmse, 'pcc': avg_pcc}

                    # 현재 조합의 점수 계산 (PCC를 최대화)
                    current_overall_score = avg_pcc
                    if current_overall_score > best_gnn_overall_score:
                        best_gnn_overall_score = current_overall_score
                        best_gnn_params = {
                            'hidden_dim': current_hidden_dim,
                            'num_heads': current_num_heads,
                            'dropout_rate': current_dropout_rate,
                            'lr': current_lr
                        }
                        # 최적의 GNN 조합일 때만 최종 예측값을 저장
                        final_best_test_predictions = np.mean(test_preds_folds_current_gnn_params, axis=0)
                        print(f"새로운 최고 성능 GNN 조합 발견: {gnn_param_key} (PCC: {best_gnn_overall_score:.4f})")
    
    # ----------------------------------------------------------------------------------------------------
    # 🔧 최종 결과 및 제출 파일 생성
    # ----------------------------------------------------------------------------------------------------
    print("\n--- GNN 하이퍼파라미터 튜닝 완료 ---")
    print("\n튜닝 결과:")
    for combo, metrics in gnn_tuning_results.items():
        print(f"GNN Params {combo}: RMSE={metrics['rmse']:.4f}, PCC={metrics['pcc']:.4f}")
    
    print(f"\n최적의 GNN 조합: {best_gnn_params} (최고 PCC: {best_gnn_overall_score:.4f})")

    # 최종 제출 파일 생성 (최적의 GNN 조합으로 계산된 예측값 사용)
    if final_best_test_predictions is not None:
        submit_df = submit_df_original.copy()
        submit_df['Inhibition'] = final_best_test_predictions
        
        # ✨✨✨ 제출 파일 인코딩 오류 해결: encoding='utf-8' 명시 ✨✨✨
        submission_filename = f"submission_APFP_N{Config.tuning_apfp_n}_TFP_N{Config.tuning_tfp_n}_GNN_H{best_gnn_params['hidden_dim']}_NH{best_gnn_params['num_heads']}_DP{best_gnn_params['dropout_rate']}_LR{best_gnn_params['lr']}.csv"
        submit_df.to_csv(submission_filename, index=False, encoding='utf-8')
        print(f"\nSubmission 파일 '{submission_filename}'이(가) UTF-8로 생성되었습니다.")
    else:
        print("최적 GNN 조합의 예측값을 찾을 수 없어 제출 파일이 생성되지 않았습니다.")