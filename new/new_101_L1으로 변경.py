# 🔧 목적: GNN (GATv2) 기반 분자 물성 예측 파이프라인 (이상치 처리 및 K-Fold 포함)

import numpy as np
import pandas as pd
import os, platform
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import rdmolops
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
from sklearn.linear_model import Lasso # <-- Lasso 임포트 추가

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
    node_feat_dim = 9
    maccs_keys_dim = 167
    morgan_dim = 2048 # 초기 Morgan Fingerprint 차원
    use_topN_morgan = 200 # Morgan Fingerprint에서 선택할 Top N 개수 (최근 튜닝 결과에 따라 200으로 고정)
    mordred_dim = 60 # Mordred 디스크립터에서 선택할 Top N 개수 (60으로 고정)

    # 이상치 처리 설정
    remove_outliers = True
    outlier_contamination = 0.02

    # Lasso alpha 값 (L1 규제 강도)
    lasso_alpha = 0.01 # <-- 새로운 하이퍼파라미터: Lasso의 alpha 값, 튜닝 대상

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
    # ----------------------------------------------------------------------------------------------------
    # 🔧 0. 하이퍼파라미터 튜닝을 위한 설정
    # ----------------------------------------------------------------------------------------------------
    # 현재는 Morgan FP 차원 200으로 고정, Mordred 60으로 고정
    # 다음 튜닝은 GNN hidden_dim, num_heads 또는 Mordred 재튜닝, Lasso alpha 튜닝 등
    
    # 예시: Lasso alpha 값 튜닝을 하고 싶다면 아래 리스트를 사용
    lasso_alphas_to_test = [0.0001, 0.001, 0.01, 0.1, 1.0] # <-- 튜닝할 alpha 값 범위
    
    best_overall_score = -float('inf')
    best_alpha = Config.lasso_alpha # 초기값
    all_results = [] # 각 alpha 값별 결과를 저장할 리스트 (alpha, score)

    # 피처 중요도 및 상관관계 시각화를 위해 필요한 변수들 초기화
    # Morgan FP 시각화에 필요
    best_morgan_selected_features_df = None
    best_lasso_model_morgan = None # <-- RandomForest 대신 Lasso 모델 저장
    best_morgan_feature_columns = None
    
    # Mordred 시각화에 필요 (현재는 Morgan 튜닝 시 Mordred 관련 변수 업데이트 안됨. 필요시 별도 튜닝 루프에서 저장)
    best_mordred_all_filtered_columns = None
    best_lasso_model_mordred = None # <-- RandomForest 대신 Lasso 모델 저장
    best_mordred_feature_columns = None


    # ----------------------------------------------------------------------------------------------------
    # 🔧 3. 데이터 로드 및 초기 준비 (한 번만 실행)
    # ----------------------------------------------------------------------------------------------------
    print("데이터 로드 및 초기 준비 시작...")
    train_df_original = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    test_df_original = pd.read_csv('./_data/dacon/new/test.csv')
    submit_df_original = pd.read_csv('./_data/dacon/new/sample_submission.csv')

    combined_smiles_raw = train_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist() + \
                          test_df_original['Canonical_Smiles'].tolist()

    print(f"총 {len(combined_smiles_raw)}개의 SMILES를 처리합니다.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 4. Mordred 디스크립터 계산 (캐싱 포함, 한 번만 실행)
    # ----------------------------------------------------------------------------------------------------
    mordred_cache_path = './_save/mordred_combined_cache.csv'
    recalculate_mordred = False

    if os.path.exists(mordred_cache_path):
        print("Mordred Descriptors 캐시 파일 로드 중...")
        try:
            mordred_combined_df = pd.read_csv(mordred_cache_path)
            if mordred_combined_df.empty or mordred_combined_df.shape[1] == 0:
                print("경고: 로드된 Mordred 캐시 파일이 비어있거나 유효하지 않아 다시 계산합니다.")
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
            mordred_combined_df = mordred_combined_df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

            num_mordred_features = mordred_combined_df.shape[1]
            print(f"\n--- Mordred 계산 결과 요약 ---")
            print(f"Mordred 계산 성공. 총 디스크립터 차원: {num_mordred_features}")
            print(f"---------------------------------------------------\n")
        except Exception as e:
            print(f"ERROR: Mordred 계산 중 오류 발생: {type(e).__name__} - {e}")
            print("Mordred 계산에 실패하여 빈 데이터프레임으로 대체합니다.")
            num_mordred_features = 0
            mordred_combined_df = pd.DataFrame(np.zeros((len(combined_smiles_raw), 0)), columns=[])
        mordred_combined_df.to_csv(mordred_cache_path, index=False)
        print("Mordred Descriptors 캐시 파일 저장 완료.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 5. Morgan Fingerprint 및 MACCS Keys 계산 (한 번만 실행)
    # ----------------------------------------------------------------------------------------------------
    print("\n글로벌 피처 (Morgan FP, MACCS Keys) 계산 시작...")
    all_morgan_fps_raw = []
    all_maccs_keys_raw = []
    for smiles in tqdm(combined_smiles_raw, desc="Morgan FP & MACCS Keys"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            all_morgan_fps_raw.append(get_morgan_fingerprint(mol))
            all_maccs_keys_raw.append(get_maccs_keys(mol))
        else:
            all_morgan_fps_raw.append(np.zeros(Config.morgan_dim, dtype=np.float32))
            all_maccs_keys_raw.append(np.zeros(Config.maccs_keys_dim, dtype=np.float32))
    all_morgan_fps_raw = np.array(all_morgan_fps_raw)
    all_maccs_keys_raw = np.array(all_maccs_keys_raw)
    print("Morgan Fingerprint 및 MACCS Keys 계산 완료.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 6. Lasso alpha 값 튜닝 루프 시작 (현재 튜닝 대상)
    # ----------------------------------------------------------------------------------------------------
    print(f"\n===== Lasso alpha 값 튜닝 시작 (테스트할 alpha: {lasso_alphas_to_test}) =====")

    for current_alpha in lasso_alphas_to_test: # <-- 루프 변수 변경
        Config.lasso_alpha = current_alpha # Config 클래스의 lasso_alpha 업데이트
        print(f"\n--- 현재 Lasso alpha: {Config.lasso_alpha}로 모델 학습 시작 ---")

        train_df = train_df_original.copy()
        test_df = test_df_original.copy()
        submit_df = submit_df_original.copy()

        all_morgan_fps = all_morgan_fps_raw.copy()
        all_maccs_keys = all_maccs_keys_raw.copy()

        # ----------------------------------------------------------------------------------------------------
        # 🔧 6.1 Mordred 특성 전처리 (Lasso 기반 피처 선택)
        # ----------------------------------------------------------------------------------------------------
        mordred_processed = None
        current_mordred_feature_columns = None

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

            mordred_filtered_cols = mordred_filtered_corr_df.columns

            if mordred_filtered_corr.shape[1] > 0:
                # RandomForestRegressor 대신 Lasso 사용
                lasso_model_mordred = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
                lasso_model_mordred.fit(mordred_filtered_corr[:len(train_df)], train_df['Inhibition'].values)

                # SelectFromModel에 Lasso 모델 전달
                selector_mordred = SelectFromModel(lasso_model_mordred, max_features=Config.mordred_dim, prefit=True, threshold=-np.inf)

                mordred_processed = selector_mordred.transform(mordred_filtered_corr)

                selected_indices = selector_mordred.get_support(indices=True)
                current_mordred_feature_columns = mordred_filtered_cols[selected_indices]
                current_mordred_all_filtered_columns = mordred_filtered_cols
            else:
                print("   - 경고: Mordred 디스크립터가 없어 Lasso 기반 피처 선택을 건너뜁니다.")
                mordred_processed = np.empty((mordred_filtered_corr.shape[0], 0))
                current_mordred_feature_columns = pd.Index([])
                current_mordred_all_filtered_columns = pd.Index([])

        print(f"Mordred 최종 처리된 데이터 차원: {mordred_processed.shape}")

        # ----------------------------------------------------------------------------------------------------
        # 🔧 6.2 Morgan Fingerprint 피처 선택 (Lasso 기반 피처 선택)
        # ----------------------------------------------------------------------------------------------------
        print(f"\nMorgan Fingerprint 피처 선택 (Top {Config.use_topN_morgan}개, Lasso 기반)...")
        if all_morgan_fps.shape[1] > 0:
            # RandomForestRegressor 대신 Lasso 사용
            lasso_model_morgan = Lasso(alpha=Config.lasso_alpha, random_state=Config.seed)
            lasso_model_morgan.fit(all_morgan_fps[:len(train_df)], train_df['Inhibition'].values)

            # SelectFromModel에 Lasso 모델 전달
            selector_morgan = SelectFromModel(lasso_model_morgan, max_features=Config.use_topN_morgan, prefit=True, threshold=-np.inf)
            morgan_fps_processed = selector_morgan.transform(all_morgan_fps)

            current_morgan_feature_columns_indices = np.where(selector_morgan.get_support())[0]
            current_morgan_feature_columns = [f'MorganFP_{i}' for i in current_morgan_feature_columns_indices]

        else:
            print("경고: Morgan Fingerprint가 0차원이므로 피처 선택을 건너뜁니다.")
            morgan_fps_processed = np.empty((all_morgan_fps.shape[0], 0))
            current_morgan_feature_columns = []
        print(f"Morgan FP 최종 처리된 데이터 차원: {morgan_fps_processed.shape}")

        # ----------------------------------------------------------------------------------------------------
        # 🔧 6.3 최종 글로벌 피처 결합
        # ----------------------------------------------------------------------------------------------------
        print("\n최종 글로벌 피처 결합...")
        global_features_combined = np.hstack([all_maccs_keys, morgan_fps_processed, mordred_processed])

        Config.global_feat_dim = global_features_combined.shape[1]
        print(f"전체 데이터에 대한 Global Feature 최종 차원: {global_features_combined.shape}")
        print(f"GNN 모델의 global_feat_dim: {Config.global_feat_dim}")


        # ----------------------------------------------------------------------------------------------------
        # 🔧 6.4 이상치(Outlier) 처리 (훈련 데이터에만 적용)
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

        test_data_list = []
        for i, smiles in tqdm(enumerate(test_df['Canonical_Smiles'].tolist()), total=len(test_df), desc="Test Graph Data"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                data = mol_to_graph_data(mol, global_feat_test[i])
                test_data_list.append(data)

        submit_data_list = []
        for i, smiles in tqdm(enumerate(test_df['Canonical_Smiles'].tolist()), total=len(test_df), desc="Submit Graph Data"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                data = mol_to_graph_data(mol, global_feat_submit[i])
                submit_data_list.append(data)
        print("그래프 데이터셋 생성 완료. 모델 훈련 준비 중...")


        # ----------------------------------------------------------------------------------------------------
        # 🔧 8. 모델 훈련 (K-Fold 교차 검증)
        # ----------------------------------------------------------------------------------------------------
        kf = KFold(n_splits=Config.k_splits, shuffle=True, random_state=Config.seed)
        oof_preds = np.zeros(len(train_data_list))
        current_fold_test_preds_list = []
        current_fold_submit_preds_list = []

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
                hidden_dim=Config.use_topN_morgan, # Morgan FP의 TopN을 hidden_dim으로 활용 (GNN hidden_dim 튜닝 시 이 부분 변경)
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

                oof_preds[val_idx] = preds

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
                    # 모델 저장 파일명에 alpha 값 추가
                    torch.save(model.state_dict(), f'./_save/gnn_model_fold_lasso_alpha_{Config.lasso_alpha}_{fold+1}.pt')
                else:
                    counter += 1
                    if counter >= Config.patience:
                        print(f"EarlyStopping at Fold {fold+1} Epoch {epoch}")
                        break

            print(f"--- Fold {fold+1} 훈련 완료. Best Score: {best_score_fold:.4f} ---")

            # 최고 성능 모델 로드 후 예측 수행
            torch_model_path = f'./_save/gnn_model_fold_lasso_alpha_{Config.lasso_alpha}_{fold+1}.pt'
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
            current_fold_test_preds_list.append(np.concatenate(fold_test_preds))

            fold_submit_preds = []
            with torch.no_grad():
                for batch in submit_loader:
                    batch = batch.to(device)
                    fold_submit_preds.append(model(batch).cpu().numpy())
            current_fold_submit_preds_list.append(np.concatenate(fold_submit_preds))

        # 현재 alpha 값에 대한 최종 OOF 예측 성능
        oof_rmse = np.sqrt(mean_squared_error(train_df['Inhibition'].values, oof_preds))
        oof_corr = pearsonr(train_df['Inhibition'].values, oof_preds)[0]
        oof_y_range = train_df['Inhibition'].values.max() - train_df['Inhibition'].values.min()
        oof_normalized_rmse = oof_rmse / oof_y_range if oof_y_range > 0 else 0
        oof_score = 0.5 * (1 - min(oof_normalized_rmse, 1)) + 0.5 * np.clip(oof_corr, 0, 1)

        print(f"\n현재 Lasso alpha ({Config.lasso_alpha})에 대한 K-Fold OOF 결과 - RMSE: {oof_rmse:.4f}, Corr: {oof_corr:.4f}, Score: {oof_score:.4f}")

        all_results.append((Config.lasso_alpha, oof_score))

        if oof_score > best_overall_score:
            best_overall_score = oof_score
            best_alpha = Config.lasso_alpha
            final_test_preds_for_best_alpha = np.mean(current_fold_test_preds_list, axis=0)
            final_submit_preds_for_best_alpha = np.mean(current_fold_submit_preds_list, axis=0)

            # 최적의 alpha 값에 대한 피처 데이터와 Lasso 모델 저장 (시각화용)
            # 현재 루프에서 생성된 current_morgan_feature_columns 및 mordred_processed (best_mordred_all_filtered_columns) 사용
            # Morgan FP는 이진이므로, Lasso coef_가 아닌 get_support()로 선택된 인덱스 기준
            best_morgan_selected_features_df = pd.DataFrame(morgan_fps_processed[:len(train_df_original)],
                                                            columns=current_morgan_feature_columns)
            best_lasso_model_morgan = lasso_model_morgan # Morgan FP 선택에 사용된 Lasso 모델
            best_morgan_feature_columns = current_morgan_feature_columns

            # Mordred는 수치형이므로 coef_ 직접 사용 가능 (하지만 여기서는 그냥 선택된 컬럼명만 저장)
            # best_lasso_model_mordred = lasso_model_mordred # Mordred 선택에 사용된 Lasso 모델
            # best_mordred_feature_columns = current_mordred_feature_columns
            # best_mordred_all_filtered_columns = current_mordred_all_filtered_columns # Pylance 경고 해결을 위해 여기에 저장


            print(f"🎉 현재까지 최고 점수 갱신! 최적 Lasso alpha: {best_alpha}, Score: {best_overall_score:.4f}")

    print("\n\n========================================================")
    print(f"Lasso alpha 튜닝 완료!")
    print(f"최적의 Lasso alpha: {best_alpha} (Score: {best_overall_score:.4f})")
    print("========================================================\n")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 9. 튜닝 결과 시각화
    # ----------------------------------------------------------------------------------------------------
    if all_results:
        alphas, scores = zip(*all_results)
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, scores, marker='o', linestyle='-', color='b')
        plt.title('Lasso Alpha 값별 OOF 성능') # <-- 제목 변경
        plt.xlabel('Lasso Alpha 값') # <-- x축 레이블 변경
        plt.ylabel('OOF Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(alphas)
        plt.tight_layout()
        plt.savefig('./_save/lasso_alpha_tuning_results.png') # <-- 저장 파일명 변경
        plt.show()
        print("Lasso alpha 튜닝 결과 시각화 완료. './_save/lasso_alpha_tuning_results.png' 저장됨.")

    # ----------------------------------------------------------------------------------------------------
    # 🔧 10. 최적 Lasso alpha에 대한 피처 중요도 시각화
    # ----------------------------------------------------------------------------------------------------
    # Lasso는 coef_ (계수)를 통해 피처 중요도를 나타냄. 계수의 절대값이 클수록 중요.

    if best_lasso_model_morgan is not None and best_morgan_feature_columns is not None:
        print(f"\n최적 Lasso alpha ({best_alpha})에 대한 Morgan FP 피처 중요도 시각화 시작...")

        # 모든 Morgan FP 비트 (2048개)에 대한 임시 컬럼 이름 생성
        all_morgan_fp_cols = [f'MorganFP_{i}' for i in range(Config.morgan_dim)]
        
        # Lasso의 coef_는 학습에 사용된 전체 피처에 대한 계수이므로, 인덱싱 필요
        feature_coefs_full = pd.Series(best_lasso_model_morgan.coef_, index=all_morgan_fp_cols)
        
        # 선택된 피처들의 계수만 필터링하고 절대값 기준 내림차순 정렬
        selected_feature_coefs = feature_coefs_full[best_morgan_feature_columns].abs().sort_values(ascending=False)

        top_n = min(20, len(selected_feature_coefs))

        if top_n > 0:
            plt.figure(figsize=(12, max(6, top_n * 0.4)))
            sns.barplot(x=selected_feature_coefs.head(top_n).values,
                        y=selected_feature_coefs.head(top_n).index)
            plt.title(f'최적 Lasso Alpha ({best_alpha}) Morgan FP 상위 {top_n}개 피처 중요도 (절대 계수)')
            plt.xlabel('피처 중요도 (절대 계수)')
            plt.ylabel('Morgan Fingerprint 비트')
            plt.tight_layout()
            plt.savefig('./_save/morgan_lasso_feature_importance.png')
            plt.show()
            print(f"Morgan FP 피처 중요도 시각화가 './_save/morgan_lasso_feature_importance.png'에 저장됨.")
        else:
            print("선택된 Morgan FP 피처가 없거나 중요도를 시각화할 수 없습니다.")
    else:
        print("피처 중요도 시각화를 위한 정보가 충분하지 않습니다. (Lasso 모델 또는 컬럼명 없음)")

    # Mordred 디스크립터에 대한 피처 중요도 시각화 (현재 루프에서는 best_lasso_model_mordred 등이 저장되지 않으므로,
    # 이 부분은 필요시 Mordred 튜닝 루프에서 활성화하거나, 별도 분석 함수로 분리하는 것이 좋습니다.)
    # if best_lasso_model_mordred is not None and best_mordred_feature_columns is not None:
    #     print(f"\n최적 Lasso alpha ({best_alpha})에 대한 Mordred 디스크립터 피처 중요도 시각화 시작...")
    #     feature_coefs_mordred = pd.Series(best_lasso_model_mordred.coef_, index=best_mordred_all_filtered_columns)
    #     selected_feature_coefs_mordred = feature_coefs_mordred[best_mordred_feature_columns].abs().sort_values(ascending=False)
    #     top_n = min(20, len(selected_feature_coefs_mordred))
    #     if top_n > 0:
    #         plt.figure(figsize=(12, max(6, top_n * 0.4)))
    #         sns.barplot(x=selected_feature_coefs_mordred.head(top_n).values,
    #                     y=selected_feature_coefs_mordred.head(top_n).index)
    #         plt.title(f'최적 Lasso Alpha ({best_alpha}) Mordred 상위 {top_n}개 피처 중요도 (절대 계수)')
    #         plt.xlabel('피처 중요도 (절대 계수)')
    #         plt.ylabel('Mordred 디스크립터')
    #         plt.tight_layout()
    #         plt.savefig('./_save/mordred_lasso_feature_importance.png')
    #         plt.show()
    #         print(f"Mordred 피처 중요도 시각화가 './_save/mordred_lasso_feature_importance.png'에 저장됨.")
    #     else:
    #         print("선택된 Mordred 피처가 없거나 중요도를 시각화할 수 없습니다.")


    # ----------------------------------------------------------------------------------------------------
    # 🔧 11. 최종 제출 파일 생성 (최적의 Lasso alpha에 대한 예측 사용)
    # ----------------------------------------------------------------------------------------------------
    print("\n최종 제출 파일 생성 중...")
    if 'final_submit_preds_for_best_alpha' in locals():
        submit_df_original['Inhibition'] = final_submit_preds_for_best_alpha
        submit_df_original.to_csv(f'./_save/submission_best_lasso_alpha_{best_alpha}.csv', index=False)
        print(f"최종 제출 파일이 './_save/submission_best_lasso_alpha_{best_alpha}.csv'에 저장되었습니다. (최적 Lasso alpha: {best_alpha})")
    else:
        print("최적의 Lasso alpha에 대한 예측값이 없어 제출 파일을 생성할 수 없습니다. (튜닝 결과가 없거나 오류 발생)")

    print("\n모든 작업 완료.")