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

# 폰트 설정 (운영체제에 따라)
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 설정 클래스
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
    use_topN_apfp = 200   # APFP에서 선택할 Top N 개수
    tfp_initial_dim = 2048 # Topological Fingerprint 원본 차원
    use_topN_tfp = 200    # TFP에서 선택할 Top N 개수

    rdkit_basic_desc_count = 5 # RDKit 기본 디스크립터 개수

    remove_outliers = True
    outlier_contamination = 0.02

    lasso_alpha = 0.1 # Lasso alpha 값

    # 캐시 파일 경로
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

# ----------------------------------------------------------------------------------------------------
# 🔧 1. 피처 추출 및 그래프 생성 함수 정의
# ----------------------------------------------------------------------------------------------------
def atom_features(atom):
    return np.array([
        atom.GetSymbol() == 'C', atom.GetSymbol() == 'N', atom.GetSymbol() == 'O',
        atom.GetSymbol() == 'S', atom.GetSymbol() == 'F', atom.GetSymbol() == 'Cl',
        atom.GetSymbol() == 'Br', atom.GetSymbol() == 'I', atom.GetDegree(),
    ], dtype=np.float32)

calc = Calculator(descriptors, ignore_3D=True)

def get_morgan_fingerprint(mol, dim=Config.morgan_dim):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim), dtype=np.float32)

def get_maccs_keys(mol):
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def get_atom_pair_fingerprint(mol):
    return np.array(Pairs.GetAtomPairFingerprintAsBitVect(mol), dtype=np.float32)

def get_topological_fingerprint(mol, dim=Config.tfp_initial_dim):
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
        edge_index = torch.tensor([[0,0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).view(1, -1)

    if y is not None:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor, y=torch.tensor(y, dtype=torch.float).view(-1))
    else:
        return Data(x=x, edge_index=edge_index, global_feat=global_feat_tensor)

# ----------------------------------------------------------------------------------------------------
# 🔧 2. GNN 모델 정의
# ----------------------------------------------------------------------------------------------------
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

# ====================================================================================
# ✅ 메인 실행 블록
# ====================================================================================
if __name__ == '__main__':
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

    combined_smiles_raw = train_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist() 

    print(f"총 {len(combined_smiles_raw)}개의 SMILES를 처리합니다.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 3. Mordred 디스크립터 계산 (캐싱 포함)
    # ----------------------------------------------------------------------------------------------------
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
        combined_mols_raw = []
        for smiles in tqdm(combined_smiles_raw, desc="SMILES -> Mol 변환"):
            mol = Chem.MolFromSmiles(smiles)
            combined_mols_raw.append(mol)
        try:
            mordred_combined_df_raw = calc.pandas(combined_mols_raw)
            mordred_combined_df = mordred_combined_df_raw.select_dtypes(include=np.number).fillna(0)
            print(f"\nMordred 계산 성공. 총 디스크립터 차원: {mordred_combined_df.shape[1]}")
        except Exception as e:
            print(f"ERROR: Mordred 계산 중 오류 발생: {type(e).__name__} - {e}")
            print("Mordred 계산에 실패하여 빈 데이터프레임으로 대체합니다.")
            mordred_combined_df = pd.DataFrame(np.zeros((len(combined_smiles_raw), 0)), columns=[])
        mordred_combined_df.to_csv(Config.mordred_cache_path, index=False)
        print("Mordred Descriptors 캐시 파일 저장 완료.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 4. 모든 Fingerprint 및 RDKit Descriptors 계산 (캐싱 포함)
    # ----------------------------------------------------------------------------------------------------
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

    combined_mols_for_fp_desc = [Chem.MolFromSmiles(s) for s in combined_smiles_raw]

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
            for i, mol in tqdm(enumerate(combined_mols_for_fp_desc), total=len(combined_mols_for_fp_desc), desc=f"Calculating {name}"):
                if mol:
                    if dim_param is not None:
                        temp_fp_list.append(gen_func(mol, dim=dim_param))
                    else:
                        temp_fp_list.append(gen_func(mol))
                else: # 유효하지 않은 SMILES 처리
                    if "Morgan" in name:
                        temp_fp_list.append(np.zeros(Config.morgan_dim, dtype=np.float32))
                    elif "MACCS" in name:
                        temp_fp_list.append(np.zeros(Config.maccs_keys_dim, dtype=np.float32))
                    elif "APFP" in name:
                        temp_fp_list.append(np.zeros(Config.apfp_target_dim, dtype=np.float32))
                    elif "TFP" in name:
                        temp_fp_list.append(np.zeros(Config.tfp_initial_dim, dtype=np.float32))
                    elif "RDKit" in name:
                        temp_fp_list.append(np.zeros(Config.rdkit_basic_desc_count, dtype=np.float32))
                    else:
                        temp_fp_list.append(np.array([], dtype=np.float32))
            
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
    # 🔧 5. 피처 전처리 및 선택 (Lasso 기반 적용)
    # ----------------------------------------------------------------------------------------------------
    train_df = train_df_original.copy()
    test_df = test_df_original.copy()
    submit_df = submit_df_original.copy()

    current_all_morgan_fps = all_morgan_fps_raw.copy()
    current_all_apfp = all_apfp_raw.copy()
    current_all_tfp = all_tfp_raw.copy()
    current_all_rdkit_descriptors = all_rdkit_descriptors_raw.copy()

    # Mordred 특성 전처리
    mordred_processed = None
    if mordred_combined_df.shape[1] == 0:
        print("경고: Mordred 디스크립터가 0차원이므로 Mordred 관련 전처리를 건너뜁니다.")
        mordred_processed = np.empty((mordred_combined_df.shape[0], 0))
    else:
        print(f"Mordred: 고정된 {Config.mordred_dim} 차원으로 피처 선택 (Lasso 기반)...")
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
            print("   - 경고: Mordred 디스크립터가 없어 Lasso 기반 피처 선택을 건너뜁니다.")
            mordred_processed = np.empty((mordred_filtered_corr.shape[0], 0))

    print(f"Mordred 최종 처리된 데이터 차원: {mordred_processed.shape}")


    # Morgan Fingerprint 피처 선택
    print(f"\nMorgan Fingerprint 피처 선택 (Top {Config.use_topN_morgan}개, Lasso 기반)...")
    morgan_fps_processed = None
    if current_all_morgan_fps.shape[1] > 0:
        lasso_model_morgan = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_morgan.fit(current_all_morgan_fps[:len(train_df)], train_df['Inhibition'].values)
        selector_morgan = SelectFromModel(lasso_model_morgan, max_features=Config.use_topN_morgan, prefit=True, threshold=-np.inf)
        morgan_fps_processed = selector_morgan.transform(current_all_morgan_fps)
    else:
        print("경고: Morgan Fingerprint가 0차원이므로 피처 선택을 건너뜁니다.")
        morgan_fps_processed = np.empty((current_all_morgan_fps.shape[0], 0))
    print(f"Morgan FP 최종 처리된 데이터 차원: {morgan_fps_processed.shape}")


    # APFP 피처 선택
    print(f"\nAPFP 피처 선택 (Top {Config.use_topN_apfp}개, Lasso 기반)...")
    apfp_processed = None
    if current_all_apfp.shape[1] > 0:
        lasso_model_apfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_apfp.fit(current_all_apfp[:len(train_df)], train_df['Inhibition'].values)
        selector_apfp = SelectFromModel(lasso_model_apfp, max_features=Config.use_topN_apfp, prefit=True, threshold=-np.inf)
        apfp_processed = selector_apfp.transform(current_all_apfp)
    else:
        print("경고: APFP가 0차원이므로 피처 선택을 건너뜁니다.")
        apfp_processed = np.empty((current_all_apfp.shape[0], 0))
    print(f"APFP 최종 처리된 데이터 차원: {apfp_processed.shape}")


    # TFP 피처 선택
    print(f"\nTFP 피처 선택 (Top {Config.use_topN_tfp}개, Lasso 기반)...")
    tfp_processed = None
    if current_all_tfp.shape[1] > 0:
        lasso_model_tfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_tfp.fit(current_all_tfp[:len(train_df)], train_df['Inhibition'].values)
        selector_tfp = SelectFromModel(lasso_model_tfp, max_features=Config.use_topN_tfp, prefit=True, threshold=-np.inf)
        tfp_processed = selector_tfp.transform(current_all_tfp)
    else:
        print("경고: TFP가 0차원이므로 피처 선택을 건너뜁니다.")
        tfp_processed = np.empty((current_all_tfp.shape[0], 0))
    print(f"TFP 최종 처리된 데이터 차원: {tfp_processed.shape}")


    # RDKit 기본 디스크립터 전처리
    print("\nRDKit 기본 디스크립터 전처리 (스케일링 & Lasso 기반)...")
    rdkit_desc_processed = None
    if current_all_rdkit_descriptors.shape[1] > 0:
        scaler_rdkit = StandardScaler()
        rdkit_scaled = scaler_rdkit.fit_transform(current_all_rdkit_descriptors)

        lasso_model_rdkit = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_rdkit.fit(rdkit_scaled[:len(train_df)], train_df['Inhibition'].values)
        selector_rdkit = SelectFromModel(lasso_model_rdkit, prefit=True, threshold=-np.inf)
        rdkit_desc_processed = selector_rdkit.transform(rdkit_scaled)
    else:
        print("경고: RDKit 기본 디스크립터가 0차원이므로 전처리를 건너뜁니다.")
        rdkit_desc_processed = np.empty((current_all_rdkit_descriptors.shape[0], 0))
    print(f"RDKit 디스크립터 최종 처리된 데이터 차원: {rdkit_desc_processed.shape}")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 5.1 최종 글로벌 피처 결합
    # ----------------------------------------------------------------------------------------------------
    print("\n최종 글로벌 피처 결합...")
    global_features_combined = np.hstack([
        all_maccs_keys_raw,
        morgan_fps_processed,
        mordred_processed,
        apfp_processed,
        tfp_processed,
        rdkit_desc_processed
    ])

    Config.global_feat_dim = global_features_combined.shape[1]
    print(f"전체 데이터에 대한 Global Feature 최종 차원: {global_features_combined.shape}")
    print(f"GNN 모델의 global_feat_dim: {Config.global_feat_dim}")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 5.2 이상치(Outlier) 처리 (훈련 데이터에만 적용)
    # ----------------------------------------------------------------------------------------------------
    if Config.remove_outliers:
        print(f"\n이상치 처리 시작 (IsolationForest, contamination={Config.outlier_contamination})...")
        train_global_features = global_features_combined[:len(train_df)]

        iso_forest = IsolationForest(n_estimators=100, contamination=Config.outlier_contamination, random_state=Config.seed, n_jobs=-1)
        outlier_preds = iso_forest.fit_predict(train_global_features)
        is_inlier = outlier_preds == 1

        original_train_len = len(train_df)
        train_df = train_df[is_inlier].reset_index(drop=True)
        global_features_train_filtered = train_global_features[is_inlier]

        print(f"   - 원본 훈련 데이터 개수: {original_train_len}")
        print(f"   - 제거된 이상치 개수: {original_train_len - len(train_df)}")
        print(f"   - 이상치 제거 후 훈련 데이터 개수: {len(train_df)}")
    else:
        global_features_train_filtered = global_features_combined[:len(train_df)]

    global_feat_train = global_features_train_filtered
    global_feat_test = global_features_combined[len(train_df_original) : len(train_df_original) + len(test_df_original)]
    global_feat_submit = global_features_combined[len(train_df_original) + len(test_df_original) : ]


    # ----------------------------------------------------------------------------------------------------
    # 🔧 6. 그래프 데이터셋 생성 및 로더 준비
    # ----------------------------------------------------------------------------------------------------
    print("\n그래프 데이터셋 생성 중...")

    train_data_list = []
    for i, smiles in tqdm(enumerate(train_df['Canonical_Smiles'].tolist()), total=len(train_df), desc="Train Graph Data"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data = mol_to_graph_data(mol, global_feat_train[i], y=train_df['Inhibition'].iloc[i])
            train_data_list.append(data)
        else:
            print(f"경고: 훈련 데이터 SMILES {smiles} (index {i}) 처리 실패. 해당 데이터 스킵.")

    test_data_list = []
    for i, smiles in tqdm(enumerate(test_df['Canonical_Smiles'].tolist()), total=len(test_df), desc="Test Graph Data"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data = mol_to_graph_data(mol, global_feat_test[i])
            test_data_list.append(data)
        else:
            print(f"경고: 테스트 데이터 SMILES {smiles} (index {i}) 처리 실패. 해당 데이터 스킵.")

    submit_data_list = []
    for i, smiles in tqdm(enumerate(test_df['Canonical_Smiles'].tolist()), total=len(test_df), desc="Submit Graph Data"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data = mol_to_graph_data(mol, global_feat_submit[i])
            submit_data_list.append(data)
        else:
            print(f"경고: 제출 데이터 SMILES {smiles} (index {i}) 처리 실패. 해당 데이터 스킵.")
    print("그래프 데이터셋 생성 완료. 모델 훈련 준비 중...")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 7. 모델 훈련 (K-Fold 교차 검증)
    # ----------------------------------------------------------------------------------------------------
    kf = KFold(n_splits=Config.k_splits, shuffle=True, random_state=Config.seed)
    oof_preds = np.zeros(len(train_data_list))
    final_test_preds = []
    final_submit_preds = []

    print(f"\nK-Fold 교차 검증 시작 (총 {Config.k_splits}개 폴드)...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_list)):
        print(f"\n--- Fold {fold+1}/{Config.k_splits} 훈련 시작 ---")

        train_subset = [train_data_list[i] for i in train_idx]
        val_subset = [train_data_list[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=Config.batch_size, shuffle=False)
        test_loader = DataLoader(test_data_list, batch_size=Config.batch_size, shuffle=False)
        submit_loader = DataLoader(submit_data_list, batch_size=Config.batch_size, shuffle=False)

        model = GATv2WithGlobal(
            node_feat_dim=Config.node_feat_dim,
            global_feat_dim=Config.global_feat_dim,
            hidden_dim=Config.use_topN_morgan, # Hidden_dim은 Morgan Top N과 연관시킴
            out_dim=1,
            num_heads=4,
            dropout_rate=0.2
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
        criterion = torch.nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=Config.lr_patience, verbose=True)

        best_score_fold = -float('inf')
        counter = 0

        for epoch in range(1, Config.epochs + 1):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs

            model.eval()
            preds, trues = [], []
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y.view(-1))
                    val_loss += loss.item() * batch.num_graphs
                    preds.append(out.cpu().numpy())
                    trues.append(batch.y.view(-1).cpu().numpy())
            preds, trues = np.concatenate(preds), np.concatenate(trues)

            for i, val_idx_single in enumerate(val_idx):
                oof_preds[val_idx_single] = preds[i]

            rmse = np.sqrt(mean_squared_error(trues, preds))
            corr = pearsonr(trues, preds)[0]
            y_range = trues.max() - trues.min()
            normalized_rmse = rmse / y_range if y_range > 0 else 0
            score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * np.clip(corr, 0, 1)

            print(f"[Fold {fold+1} Epoch {epoch}] Train Loss: {train_loss/len(train_subset):.4f}, Val Loss: {val_loss/len(val_subset):.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}, Score: {score:.4f}")

            scheduler.step(val_loss/len(val_subset))

            if score > best_score_fold:
                best_score_fold = score
                counter = 0
                torch.save(model.state_dict(), f'./_save/gnn_model_fold_{fold+1}.pt')
            else:
                counter += 1
                if counter >= Config.patience:
                    print(f"EarlyStopping at Fold {fold+1} Epoch {epoch}")
                    break

        print(f"--- Fold {fold+1} 훈련 완료. Best Score: {best_score_fold:.4f} ---")

        torch_model_path = f'./_save/gnn_model_fold_{fold+1}.pt'
        if os.path.exists(torch_model_path):
            model.load_state_dict(torch.load(torch_model_path))
        else:
            print(f"경고: {torch_model_path} 모델 파일이 존재하지 않아 해당 폴드의 예측을 건너뜀.")
            continue

        model.eval()

        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                fold_test_preds.append(model(batch).cpu().numpy())
        final_test_preds.append(np.concatenate(fold_test_preds))

        fold_submit_preds = []
        with torch.no_grad():
            for batch in submit_loader:
                batch = batch.to(device)
                fold_submit_preds.append(model(batch).cpu().numpy())
        final_submit_preds.append(np.concatenate(fold_submit_preds))

    oof_rmse = np.sqrt(mean_squared_error(train_df['Inhibition'].values, oof_preds))
    oof_corr = pearsonr(train_df['Inhibition'].values, oof_preds)[0]
    oof_y_range = train_df['Inhibition'].values.max() - train_df['Inhibition'].values.min()
    oof_normalized_rmse = oof_rmse / oof_y_range if oof_y_range > 0 else 0
    oof_score = 0.5 * (1 - min(oof_normalized_rmse, 1)) + 0.5 * np.clip(oof_corr, 0, 1)

    print(f"\n========================================================")
    print(f"K-Fold 전체 OOF 결과 - RMSE: {oof_rmse:.4f}, Corr: {oof_corr:.4f}, Score: {oof_score:.4f}")
    print("========================================================\n")

    # ----------------------------------------------------------------------------------------------------
    # 8. 최종 제출 파일 생성 (K-Fold 예측 평균)
    # ----------------------------------------------------------------------------------------------------
    print("\n최종 제출 파일 생성 중...")
    if final_test_preds and final_submit_preds:
        avg_test_preds = np.mean(final_test_preds, axis=0)
        avg_submit_preds = np.mean(final_submit_preds, axis=0)

        submit_df_original['Inhibition'] = avg_submit_preds
        
        submission_filename = f'./_save/submission_combined_features_LassoAlpha_{Config.lasso_alpha}_APFP_TFP_TopN.csv'
        submit_df_original.to_csv(submission_filename, index=False)
        print(f"최종 제출 파일이 '{submission_filename}'에 저장되었습니다.")
    else:
        print("최종 예측값이 없어 제출 파일을 생성할 수 없습니다. (K-Fold 훈련이 실패했거나 데이터 문제)")

    print("\n모든 작업 완료.")