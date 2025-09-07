# 🔧 목적: GNN (GATv2) 기반 분자 물성 예측 파이프라인 (이상치 처리 및 K-Fold 포함)
#         - RDKit 기본 디스크립터, APFP, TFP 추가
#         - 모든 특성 선택에 Lasso (L1 규제) 기반 적용
#         - APFP 및 TFP 계산 결과 캐싱 추가

import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
import pickle # 캐시 저장/로드를 위해 pickle 사용 (numpy 배열 저장에 용이)

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem # AllChem에서 GetMorganFingerprintAsBitVect 등을 사용
from rdkit.Chem.AtomPairs import Pairs # GetAtomPairFingerprintAsBitVect를 위해 임포트
from rdkit.Chem import RDKFingerprint # TFP 임포트
from mordred import Calculator, descriptors
import torch
from torch.nn import Linear, Module, ReLU, Dropout, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    # Feature Dimensions (여기에 기본값을 설정하고 루프에서 업데이트)
    node_feat_dim = 9 # 원자 피처 (고정)

    maccs_keys_dim = 167 # MACCS Keys 차원 (고정)
    morgan_dim = 2048 # Morgan Fingerprint 원본 차원 (생성 시)
    use_topN_morgan = 200 # Morgan Fingerprint에서 선택할 Top N 개수 (최신 결과에 따라 200으로 고정)
    mordred_dim = 60 # Mordred 디스크립터에서 선택할 Top N 개수 (최신 결과에 따라 60으로 고정)

    # 새로 추가될 Fingerprint 및 RDKit Descriptors 차원 정의
    # APFP는 GetAtomPairFingerprintAsBitVect가 nBits를 받지 않으므로, 이 값은 현재 사용되지 않습니다.
    # 그러나 통일된 차원으로 패딩하기 위한 목표 차원으로 Morgan FP의 차원을 재활용할 수 있습니다.
    apfp_target_dim = 2048 # APFP를 패딩할 목표 차원 (Morgan FP와 동일하게 설정)
    tfp_initial_dim = 2048 # Topological Fingerprint 원본 차원 (RDKFingerprint의 fpSize에 사용)

    # RDKit 기본 디스크립터 개수 (LogP, TPSA, MolWt, NumHDonors, NumHAcceptors)
    rdkit_basic_desc_count = 5

    # 이상치 처리 설정
    remove_outliers = True
    outlier_contamination = 0.02

    # Lasso alpha 값 (L1 규제 강도)
    lasso_alpha = 0.1 # <-- 최적 튜닝 결과에 따라 0.1로 고정

    # 캐시 파일 경로 (추가)
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
    """원자 특성 벡터를 생성합니다."""
    return np.array([
        atom.GetSymbol() == 'C',
        atom.GetSymbol() == 'N',
        atom.GetSymbol() == 'O',
        atom.GetSymbol() == 'S',
        atom.GetSymbol() == 'F',
        atom.GetSymbol() == 'Cl',
        atom.GetSymbol() == 'Br',
        atom.GetSymbol() == 'I',
        atom.GetDegree(),
    ], dtype=np.float32)

# Mordred 계산기 초기화
calc = Calculator(descriptors, ignore_3D=True)

def get_morgan_fingerprint(mol, dim=Config.morgan_dim):
    """Morgan Fingerprint를 생성합니다."""
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim), dtype=np.float32)

def get_maccs_keys(mol):
    """MACCS Keys를 생성합니다."""
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def get_atom_pair_fingerprint(mol):
    """Atom Pair Fingerprint를 생성합니다."""
    return np.array(Pairs.GetAtomPairFingerprintAsBitVect(mol), dtype=np.float32)

def get_topological_fingerprint(mol, dim=Config.tfp_initial_dim):
    """Topological Fingerprint (RDKitFP)를 생성합니다."""
    return np.array(RDKFingerprint(mol, fpSize=dim), dtype=np.float32)

def get_rdkit_descriptors(mol):
    """RDKit 기본 디스크립터 (LogP, TPSA, MolWt, NumHDonors, NumHAcceptors)를 계산합니다."""
    if mol is None:
        return np.array([0.0] * Config.rdkit_basic_desc_count, dtype=np.float32)

    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    return np.array([logp, tpsa, mw, hbd, hba], dtype=np.float32)

# APFP처럼 가변 길이 배열을 고정 길이로 통일하는 함수
def standardize_fingerprints(fp_list, target_dim, dtype=np.float32):
    """
    가변 길이 핑거프린트 리스트를 특정 차원으로 패딩하거나 자릅니다.
    주로 APFP와 같이 길이가 가변적인 핑거프린트를 고정 길이로 만들 때 사용합니다.
    """
    if not fp_list:
        return np.empty((0, target_dim), dtype=dtype)

    standardized_fps = []
    for fp in fp_list:
        if fp.size == 0:
            standardized_fps.append(np.zeros(target_dim, dtype=dtype))
        elif fp.shape[0] < target_dim:
            padded_fp = np.pad(fp, (0, target_dim - fp.shape[0]), 'constant')
            standardized_fps.append(padded_fp)
        else: # fp.shape[0] >= target_dim
            standardized_fps.append(fp[:target_dim]) # 목표 차원보다 길면 자르기
    return np.array(standardized_fps, dtype=dtype)


def mol_to_graph_data(mol, global_feat, y=None):
    """RDKit Mol 객체와 글로벌 피처를 PyTorch Geometric Data 객체로 변환합니다."""
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
        edge_index = torch.tensor([[0,0]], dtype=torch.long) # 빈 분자 그래프 처리
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
    # ----------------------------------------------------------------------------------------------------
    # 🔧 0. 하이퍼파라미터 설정 (고정된 값 사용)
    # ----------------------------------------------------------------------------------------------------
    # 현재 Morgan FP 차원 200, Mordred 60, Lasso Alpha 0.1로 고정
    # 다음 튜닝은 GNN hidden_dim, num_heads, 레이어 수, dropout_rate, outlier_contamination 등
    
    # 피처 중요도 및 상관관계 시각화를 위해 필요한 변수들 초기화 (최적 alpha에 대한 것만 저장)
    best_morgan_selected_features_df = None
    best_lasso_model_morgan = None
    best_morgan_feature_columns = None
    
    best_mordred_feature_columns = None
    best_lasso_model_mordred = None
    best_mordred_all_filtered_columns = pd.Index([]) # Pylance 경고 해결 위해 변수명 통일 (사용 여부와 무관하게 정의)


    # ----------------------------------------------------------------------------------------------------
    # 🔧 3. 데이터 로드 및 초기 준비 (한 번만 실행)
    # ----------------------------------------------------------------------------------------------------
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

    combined_smiles_raw = train_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist() # 제출용 데이터도 포함

    print(f"총 {len(combined_smiles_raw)}개의 SMILES를 처리합니다.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 4. Mordred 디스크립터 계산 (캐싱 포함, 한 번만 실행)
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
            
            num_mordred_features = mordred_combined_df.shape[1]
            print(f"\n--- Mordred 계산 결과 요약 ---")
            print(f"Mordred 계산 성공. 총 디스크립터 차원: {num_mordred_features}")
            print(f"---------------------------------------------------\n")
        except Exception as e:
            print(f"ERROR: Mordred 계산 중 오류 발생: {type(e).__name__} - {e}")
            print("Mordred 계산에 실패하여 빈 데이터프레임으로 대체합니다.")
            num_mordred_features = 0
            mordred_combined_df = pd.DataFrame(np.zeros((len(combined_smiles_raw), 0)), columns=[])
        mordred_combined_df.to_csv(Config.mordred_cache_path, index=False)
        print("Mordred Descriptors 캐시 파일 저장 완료.")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 5. 모든 Fingerprint 및 RDKit Descriptors 계산 (캐싱 포함)
    # ----------------------------------------------------------------------------------------------------
    print("\n모든 Fingerprint 및 RDKit Descriptors 처리 중 (캐싱 활용)...")

    # 모든 FP/Desc를 저장할 컨테이너 초기화 (리스트)
    calculated_morgan_fps = []
    calculated_maccs_keys = []
    calculated_apfp = []
    calculated_tfp = []
    calculated_rdkit_descriptors = []

    # 각 핑거프린트/디스크립터별 캐싱 로직
    fp_desc_configs = [
        {"name": "Morgan FP", "path": Config.morgan_cache_path, "gen_func": get_morgan_fingerprint, "target_list": calculated_morgan_fps, "dim_param": Config.morgan_dim},
        {"name": "MACCS Keys", "path": Config.maccs_cache_path, "gen_func": get_maccs_keys, "target_list": calculated_maccs_keys, "dim_param": None}, # MACCS는 고정 dim
        {"name": "APFP", "path": Config.apfp_cache_path, "gen_func": get_atom_pair_fingerprint, "target_list": calculated_apfp, "dim_param": None}, # APFP는 nBits 없음
        {"name": "TFP", "path": Config.tfp_cache_path, "gen_func": get_topological_fingerprint, "target_list": calculated_tfp, "dim_param": Config.tfp_initial_dim},
        {"name": "RDKit Descriptors", "path": Config.rdkit_desc_cache_path, "gen_func": get_rdkit_descriptors, "target_list": calculated_rdkit_descriptors, "dim_param": None} # RDKit desc는 고정 개수
    ]

    # 임시 몰 객체 리스트 (계산 필요 시에만 생성)
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
                    target_list.extend(list(loaded_data)) # 로드된 데이터를 리스트에 추가
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
                    # dim_param이 있으면 전달, 없으면 gen_func만 호출
                    if dim_param is not None:
                        temp_fp_list.append(gen_func(mol, dim=dim_param))
                    else:
                        temp_fp_list.append(gen_func(mol))
                else:
                    # 유효하지 않은 SMILES에 대한 처리
                    if "Morgan" in name:
                        temp_fp_list.append(np.zeros(Config.morgan_dim, dtype=np.float32))
                    elif "MACCS" in name:
                        temp_fp_list.append(np.zeros(Config.maccs_keys_dim, dtype=np.float32))
                    elif "APFP" in name:
                        temp_fp_list.append(np.zeros(Config.apfp_target_dim, dtype=np.float32)) # APFP는 통일된 차원으로 0 채움
                    elif "TFP" in name:
                        temp_fp_list.append(np.zeros(Config.tfp_initial_dim, dtype=np.float32))
                    elif "RDKit" in name:
                        temp_fp_list.append(np.zeros(Config.rdkit_basic_desc_count, dtype=np.float32))
                    else: # 안전장치
                        temp_fp_list.append(np.array([], dtype=np.float32))
            
            # APFP는 특별히 차원 통일 함수를 거쳐야 함
            if name == "APFP":
                processed_fps = standardize_fingerprints(temp_fp_list, Config.apfp_target_dim)
                target_list.extend(list(processed_fps)) # extend는 리스트에 원소를 추가
                np.save(path, processed_fps)
            else:
                arr = np.array(temp_fp_list)
                target_list.extend(list(arr))
                np.save(path, arr)
            print(f"{name} 캐시 파일 저장 완료. 차원: {target_list[0].shape if target_list else 'N/A'}")
        
    # 리스트를 numpy 배열로 변환 (모든 FP/Desc가 이제 target_list에 채워져 있음)
    all_morgan_fps_raw = np.array(calculated_morgan_fps)
    all_maccs_keys_raw = np.array(calculated_maccs_keys)
    all_apfp_raw = np.array(calculated_apfp) # 이미 standardize_fingerprints를 통해 통일됨
    all_tfp_raw = np.array(calculated_tfp)
    all_rdkit_descriptors_raw = np.array(calculated_rdkit_descriptors)


    print("모든 Fingerprint 및 RDKit Descriptors 계산 또는 로드 완료.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 6. 피처 전처리 및 선택 (Lasso 기반 적용)
    # ----------------------------------------------------------------------------------------------------
    train_df = train_df_original.copy()
    test_df = test_df_original.copy()
    submit_df = submit_df_original.copy()

    # 스케일링을 위해 데이터 복사 (원본 데이터 보존)
    current_all_morgan_fps = all_morgan_fps_raw.copy()
    current_all_apfp = all_apfp_raw.copy()
    current_all_tfp = all_tfp_raw.copy()
    current_all_rdkit_descriptors = all_rdkit_descriptors_raw.copy()


    # 6.1 Mordred 특성 전처리 (Lasso 기반 피처 선택)
    mordred_processed = None
    if mordred_combined_df.shape[1] == 0:
        print("경고: Mordred 디스크립터가 0차원이므로 Mordred 관련 전처리를 건너뜁니다.")
        mordred_processed = np.empty((mordred_combined_df.shape[0], 0))
        best_mordred_feature_columns = pd.Index([]) # 변수 정의
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

        mordred_filtered_cols = mordred_filtered_corr_df.columns
        best_mordred_all_filtered_columns = mordred_filtered_cols

        if mordred_filtered_corr.shape[1] > 0:
            lasso_model_mordred = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
            lasso_model_mordred.fit(mordred_filtered_corr[:len(train_df)], train_df['Inhibition'].values)
            selector_mordred = SelectFromModel(lasso_model_mordred, max_features=Config.mordred_dim, prefit=True, threshold=-np.inf)
            mordred_processed = selector_mordred.transform(mordred_filtered_corr)

            selected_indices_mordred = selector_mordred.get_support(indices=True)
            best_mordred_feature_columns = mordred_filtered_cols[selected_indices_mordred]
            best_lasso_model_mordred = lasso_model_mordred
        else:
            print("   - 경고: Mordred 디스크립터가 없어 Lasso 기반 피처 선택을 건너뜁니다.")
            mordred_processed = np.empty((mordred_filtered_corr.shape[0], 0))
            best_mordred_feature_columns = pd.Index([])


    print(f"Mordred 최종 처리된 데이터 차원: {mordred_processed.shape}")


    # 6.2 Morgan Fingerprint 피처 선택 (Lasso 기반 피처 선택)
    print(f"\nMorgan Fingerprint 피처 선택 (Top {Config.use_topN_morgan}개, Lasso 기반)...")
    morgan_fps_processed = None
    if current_all_morgan_fps.shape[1] > 0:
        lasso_model_morgan = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_morgan.fit(current_all_morgan_fps[:len(train_df)], train_df['Inhibition'].values)
        selector_morgan = SelectFromModel(lasso_model_morgan, max_features=Config.use_topN_morgan, prefit=True, threshold=-np.inf)
        morgan_fps_processed = selector_morgan.transform(current_all_morgan_fps)

        current_morgan_feature_columns_indices = np.where(selector_morgan.get_support())[0]
        best_morgan_feature_columns = [f'MorganFP_{i}' for i in current_morgan_feature_columns_indices]
        best_lasso_model_morgan = lasso_model_morgan
    else:
        print("경고: Morgan Fingerprint가 0차원이므로 피처 선택을 건너뜁니다.")
        morgan_fps_processed = np.empty((current_all_morgan_fps.shape[0], 0))
        best_morgan_feature_columns = []
    print(f"Morgan FP 최종 처리된 데이터 차원: {morgan_fps_processed.shape}")


    # 6.3 APFP 피처 선택 (Lasso 기반 피처 선택)
    print(f"\nAPFP 피처 선택 (Lasso 기반)...")
    apfp_processed = None
    if current_all_apfp.shape[1] > 0:
        lasso_model_apfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_apfp.fit(current_all_apfp[:len(train_df)], train_df['Inhibition'].values)
        selector_apfp = SelectFromModel(lasso_model_apfp, prefit=True, threshold=-np.inf) # 모든 피처를 포함
        apfp_processed = selector_apfp.transform(current_all_apfp)
    else:
        print("경고: APFP가 0차원이므로 피처 선택을 건너뜁니다.")
        apfp_processed = np.empty((current_all_apfp.shape[0], 0))
    print(f"APFP 최종 처리된 데이터 차원: {apfp_processed.shape}")


    # 6.4 TFP 피처 선택 (Lasso 기반 피처 선택)
    print(f"\nTFP 피처 선택 (Lasso 기반)...")
    tfp_processed = None
    if current_all_tfp.shape[1] > 0:
        lasso_model_tfp = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_tfp.fit(current_all_tfp[:len(train_df)], train_df['Inhibition'].values)
        selector_tfp = SelectFromModel(lasso_model_tfp, prefit=True, threshold=-np.inf)
        tfp_processed = selector_tfp.transform(current_all_tfp)
    else:
        print("경고: TFP가 0차원이므로 피처 선택을 건너뜁니다.")
        tfp_processed = np.empty((current_all_tfp.shape[0], 0))
    print(f"TFP 최종 처리된 데이터 차원: {tfp_processed.shape}")


    # 6.5 RDKit 기본 디스크립터 전처리 (스케일링 & Lasso 기반 선택)
    print("\nRDKit 기본 디스크립터 전처리 (스케일링 & Lasso 기반)...")
    rdkit_desc_processed = None
    if current_all_rdkit_descriptors.shape[1] > 0:
        scaler_rdkit = StandardScaler()
        rdkit_scaled = scaler_rdkit.fit_transform(current_all_rdkit_descriptors)

        lasso_model_rdkit = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
        lasso_model_rdkit.fit(rdkit_scaled[:len(train_df)], train_df['Inhibition'].values)
        selector_rdkit = SelectFromModel(lasso_model_rdkit, prefit=True, threshold=-np.inf) # 모든 피처를 포함
        rdkit_desc_processed = selector_rdkit.transform(rdkit_scaled)
    else:
        print("경고: RDKit 기본 디스크립터가 0차원이므로 전처리를 건너뜁니다.")
        rdkit_desc_processed = np.empty((current_all_rdkit_descriptors.shape[0], 0))
    print(f"RDKit 디스크립터 최종 처리된 데이터 차원: {rdkit_desc_processed.shape}")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 6.6 최종 글로벌 피처 결합
    # ----------------------------------------------------------------------------------------------------
    print("\n최종 글로벌 피처 결합...")
    global_features_combined = np.hstack([
        all_maccs_keys_raw,
        morgan_fps_processed,
        mordred_processed,
        apfp_processed, # 이제 차원이 통일되고 캐싱됨
        tfp_processed, # 이제 캐싱됨
        rdkit_desc_processed # 이제 캐싱됨
    ])

    Config.global_feat_dim = global_features_combined.shape[1]
    print(f"전체 데이터에 대한 Global Feature 최종 차원: {global_features_combined.shape}")
    print(f"GNN 모델의 global_feat_dim: {Config.global_feat_dim}")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 6.7 이상치(Outlier) 처리 (훈련 데이터에만 적용)
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
    # 🔧 7. 그래프 데이터셋 생성 및 로더 준비
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
    # 🔧 8. 모델 훈련 (K-Fold 교차 검증)
    #-----------------------------------------------------------------------------------------------------
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
            hidden_dim=Config.use_topN_morgan,
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
    # 🔧 9. 피처 중요도 시각화 (Morgan FP만)
    # ----------------------------------------------------------------------------------------------------
    if best_lasso_model_morgan is not None and best_morgan_feature_columns is not None:
        print(f"\nMorgan FP 피처 중요도 시각화 시작 (Lasso Alpha: {Config.lasso_alpha})...")

        all_morgan_fp_cols = [f'MorganFP_{i}' for i in range(Config.morgan_dim)]
        feature_coefs_full = pd.Series(best_lasso_model_morgan.coef_, index=all_morgan_fp_cols)
        
        selected_feature_coefs = feature_coefs_full[best_morgan_feature_columns].abs().sort_values(ascending=False)

        top_n = min(20, len(selected_feature_coefs))

        if top_n > 0:
            plt.figure(figsize=(12, max(6, top_n * 0.4)))
            sns.barplot(x=selected_feature_coefs.head(top_n).values,
                        y=selected_feature_coefs.head(top_n).index)
            plt.title(f'Morgan FP 상위 {top_n}개 피처 중요도 (Lasso 절대 계수)')
            plt.xlabel('피처 중요도 (절대 계수)')
            plt.ylabel('Morgan Fingerprint 비트')
            plt.tight_layout()
            plt.savefig('./_save/morgan_lasso_feature_importance.png')
            plt.show()
            print(f"Morgan FP 피처 중요도 시각화가 './_save/morgan_lasso_feature_importance.png'에 저장됨.")
        else:
            print("선택된 Morgan FP 피처가 없거나 중요도를 시각화할 수 없습니다.")
    else:
        print("Morgan FP 피처 중요도 시각화를 위한 정보가 충분하지 않습니다. (Lasso 모델 또는 컬럼명 없음)")

    # ----------------------------------------------------------------------------------------------------
    # 10. 최종 제출 파일 생성 (K-Fold 예측 평균)
    # ----------------------------------------------------------------------------------------------------
    print("\n최종 제출 파일 생성 중...")
    if final_test_preds and final_submit_preds:
        avg_test_preds = np.mean(final_test_preds, axis=0)
        avg_submit_preds = np.mean(final_submit_preds, axis=0)

        submit_df_original['Inhibition'] = avg_submit_preds
        
        submission_filename = f'./_save/submission_combined_features_LassoAlpha_{Config.lasso_alpha}.csv'
        submit_df_original.to_csv(submission_filename, index=False)
        print(f"최종 제출 파일이 '{submission_filename}'에 저장되었습니다.")
    else:
        print("최종 예측값이 없어 제출 파일을 생성할 수 없습니다. (K-Fold 훈련이 실패했거나 데이터 문제)")

    print("\n모든 작업 완료.")