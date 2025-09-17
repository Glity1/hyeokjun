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
from scipy.stats import pearsonr # PCC 계산용
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class Config:
    seed = 42 # 초기 기본 시드 (각 런마다 이 값을 덮어씁니다.)
    epochs = 200
    patience = 20
    lr_patience = 10
    batch_size = 64
    k_splits = 5

    # ✨✨✨ 앙상블에 사용할 여러 랜덤 시드 값들 ✨✨✨
    run_seeds = [42, 15, 222, 123, 777] # 5개의 시드를 사용하여 앙상블 예시

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

    # ✨✨✨ 최적으로 조정된 GNN 하이퍼파라미터를 Config에 직접 고정 ✨✨✨
    # (이전 튜닝 결과를 바탕으로 여기에 단일 값을 넣어주세요.)
    hidden_dim = 128     # 예시 값
    num_heads = 4        # 예시 값
    dropout_rate = 0.2   # 예시 값
    lr = 0.001           # 예시 값

    mordred_cache_path = './_save/mordred_combined_cache.csv'
    morgan_cache_path = './_save/morgan_fps_cache.npy'
    maccs_cache_path = './_save/maccs_keys_cache.npy'
    apfp_cache_path = './_save/apfp_cache.npy'
    tfp_cache_path = './_save/tfp_cache.npy'
    rdkit_desc_cache_path = './_save/rdkit_desc_cache.npy'

# 시드 고정 함수 (각 런마다 호출)
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

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
        if fp.size == 0:
            standardized_fps.append(np.zeros(target_dim, dtype=dtype))
        elif fp.shape[0] < target_dim:
            padded_fp = np.pad(fp, (0, target_dim - fp.shape[0]), 'constant')
            standardized_fps.append(padded_fp)
        else:
            standardized_fps.append(fp[:target_dim])
    return np.array(standardized_fps, dtype=dtype)

def mol_to_graph_data(mol, global_feat, y=None):
    if mol is None:
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
        if x.size(0) > 0:
            edge_index = torch.tensor([[0,0]], dtype=torch.long)
        else:
            return None
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

# ✨✨✨ 새로운 평가 산식 'Score' 계산 함수 정의 ✨✨✨
def calculate_competition_score(y_true, y_pred):
    # PCC 계산
    pcc = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    
    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # A = Normalized RMSE 오차. Inhibition(%)의 범위가 0~100이므로, RMSE를 100으로 나누어 정규화
    # A가 1보다 클 경우 1로 클리핑하여 (1 - min(A, 1)) 항이 음수가 되지 않도록 함
    A = rmse / 100.0 
    
    # Score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    score = 0.5 * (1 - min(A, 1)) + 0.5 * pcc
    
    return score, rmse, pcc # Score와 함께 원래의 RMSE, PCC도 반환하여 로그에 사용

if __name__ == '__main__':
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

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
        combined_mols_raw_for_mordred = []
        for smiles in tqdm(combined_smiles_raw, desc="SMILES -> Mol 변환 (Mordred)"):
            mol = Chem.MolFromSmiles(smiles)
            combined_mols_raw_for_mordred.append(mol)

        try:
            mordred_df_temp = calc.pandas(combined_mols_raw_for_mordred, n_jobs=-1, quiet=True)
            
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
                mol = Chem.MolFromSmiles(smiles)
                if dim_param is not None:
                    temp_fp_list.append(gen_func(mol, dim=dim_param))
                else:
                    temp_fp_list.append(gen_func(mol))

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

    X_train_global = global_features_combined[:len(train_df_original)]
    y_train_original = train_df_original['Inhibition'].values

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
    
    X_test_global = global_features_combined[len(train_df_original):]
    
    # ✨✨✨ 시드별 앙상블을 위한 외부 루프 시작 ✨✨✨
    all_runs_final_predictions = [] # 각 시드에서 나온 최종 테스트 예측값을 저장할 리스트

    print(f"\n--- 총 {len(Config.run_seeds)}개의 시드로 앙상블 학습 시작 ---")

    for run_idx, current_seed in enumerate(Config.run_seeds):
        print(f"\n--- 앙상블 런 {run_idx+1}/{len(Config.run_seeds)} (Current Seed: {current_seed}) ---")
        
        # 현재 런의 시드 설정
        set_seed(current_seed)
        
        kf = KFold(n_splits=Config.k_splits, shuffle=True, random_state=Config.seed)
        test_predictions_folds = [] # 현재 시드에서 K-Fold 예측값을 저장할 리스트

        print("\n--- K-Fold 교차 검증 및 모델 학습 시작 ---")

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

            # 모델 초기화 (Config에 고정된 하이퍼파라미터 사용)
            model = GATv2WithGlobal(
                node_feat_dim=Config.node_feat_dim,
                global_feat_dim=global_features_combined.shape[1],
                hidden_dim=Config.hidden_dim, 
                out_dim=1,
                num_heads=Config.num_heads, 
                dropout_rate=Config.dropout_rate 
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr) 
            criterion = torch.nn.MSELoss()
            # ✨✨✨ 스케줄러 기준을 새로운 'Score'로 변경 (높을수록 좋으므로 mode='max') ✨✨✨
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=Config.lr_patience, verbose=True)

            best_val_score = -np.inf # Score는 클수록 좋으므로 초기값을 음의 무한대로 설정
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
                
                current_val_score, val_rmse, val_pcc = calculate_competition_score(np.array(val_targets), np.array(val_preds))

                scheduler.step(current_val_score)

                # Early stopping
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    epochs_no_improve = 0
                    # Best 모델 저장 (특정 폴드 및 현재 시드, 고정된 GNN 파라미터 조합명으로)
                    torch.save(model.state_dict(), 
                               f'./_save/best_model_fold_{fold}_seed{Config.seed}_h{Config.hidden_dim}_nh{Config.num_heads}_dp{Config.dropout_rate}_lr{Config.lr}.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == Config.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0 or epoch == 1 or epoch == Config.epochs:
                    print(f"Epoch {epoch}/{Config.epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, Val Score: {current_val_score:.4f}")

            # Load best model for test prediction for this fold
            model.load_state_dict(torch.load(
                f'./_save/best_model_fold_{fold}_seed{Config.seed}_h{Config.hidden_dim}_nh{Config.num_heads}_dp{Config.dropout_rate}_lr{Config.lr}.pth'))
            model.eval()

            final_val_preds = []
            final_val_targets = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    final_val_preds.extend(out.cpu().numpy())
                    final_val_targets.extend(data.y.cpu().numpy())
            
            final_val_score, final_val_rmse, final_val_pcc = calculate_competition_score(np.array(final_val_targets), np.array(final_val_preds))
            print(f"Fold {fold+1} Best Val Score: {final_val_score:.4f} (Corresponding RMSE: {final_val_rmse:.4f}, PCC: {final_val_pcc:.4f})")

            # Test prediction for this fold
            test_data_list = []
            for i in tqdm(range(len(test_df_original)), desc="Test Graph Data"):
                smiles = test_df_original.iloc[i]['Canonical_Smiles']
                mol = Chem.MolFromSmiles(smiles)
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
            test_predictions_folds.append(np.array(fold_test_preds))

        # 현재 시드에 대한 K-Fold 예측값 평균
        current_seed_final_preds = np.mean(test_predictions_folds, axis=0)
        all_runs_final_predictions.append(current_seed_final_preds)

    print("\n--- 모든 앙상블 런 완료. 최종 예측값 평균 계산 중 ---")
    final_preds_mean = np.mean(all_runs_final_predictions, axis=0)

    # 0 ~ 100 범위로 클리핑
    final_preds_clipped = np.clip(final_preds_mean, 0, 100)

    # submission DataFrame 생성
    submit_df = submit_df_original.copy()
    submit_df['Inhibition'] = final_preds_clipped
    
    # 제출 파일명 설정 (앙상블 정보 및 GNN 하이퍼파라미터 포함)
    num_runs = len(Config.run_seeds)
    submission_filename = (f"submission_ENSEMBLE_{num_runs}Runs"
                           f"_COMP_SCORE_Optimized_GNN_H{Config.hidden_dim}_NH{Config.num_heads}"
                           f"_DP{Config.dropout_rate}_LR{Config.lr}.csv") # 파일명에 PCC 최적화 명시

    # CSV 파일로 저장 (인코딩 및 인덱스 설정 중요)
    submit_df.to_csv(f'./_save/{submission_filename}', index=False, encoding='utf-8')

    print(f"\n최종 앙상블 Submission 파일 '{f'./_save/{submission_filename}'}'이(가) UTF-8로 성공적으로 생성되었습니다.")