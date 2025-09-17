import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# 설정
os.makedirs('./_save', exist_ok=True)
seed = 42
np.random.seed(seed)

# ✅ 데이터 로드
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
smiles_list = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values

# ✅ RDKit + MACCS 계산 함수
rdkit_desc_list = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, Descriptors.RingCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAliphaticRings,
    Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.ExactMolWt,
    Descriptors.MolMR, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge
]

def get_rdkit_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rdkit_desc = [f(mol) for f in rdkit_desc_list]
        rdkit_desc = np.nan_to_num(rdkit_desc).astype(np.float32)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        return np.concatenate([rdkit_desc, maccs_arr.astype(np.float32)])
    except:
        return None

# ✅ Morgan Fingerprint 계산 함수
def get_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ✅ 전체 특성 계산
rdkit_maccs_list = []
morgan_fp_list = []

for s in smiles_list:
    rdkit_maccs = get_rdkit_maccs(s)
    morgan_fp = get_morgan_fp(s)
    if rdkit_maccs is not None and morgan_fp is not None:
        rdkit_maccs_list.append(rdkit_maccs)
        morgan_fp_list.append(morgan_fp)

X_rdkit_maccs = np.array(rdkit_maccs_list)  # (n, 187)
X_morgan_full = np.array(morgan_fp_list)    # (n, 2048)
y = y[:len(X_rdkit_maccs)]

# ✅ Morgan Top150 불러오기
morgan_top150_idx = np.load('./_save/top150_morgan_indices.npy')
X_morgan150 = X_morgan_full[:, morgan_top150_idx]

# ✅ 전체 조합
X_combined = np.concatenate([X_rdkit_maccs, X_morgan150], axis=1)

# ✅ 전체 조합 기준 SelectFromModel 실행
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_combined, y)
selector = SelectFromModel(rf, threshold=-np.inf, max_features=50, prefit=True)
X_selected = selector.transform(X_combined)
selected_mask = selector.get_support()

# ✅ 선택된 전체 특성 중 Morgan150 위치 확인
morgan_start_idx = X_rdkit_maccs.shape[1]  # 187
morgan_selected_indices = [i - morgan_start_idx for i, is_sel in enumerate(selected_mask[morgan_start_idx:]) if is_sel]

# ✅ Morgan150 내 중요도 높은 Top30, Top50 추출
morgan_importances = rf.feature_importances_[morgan_start_idx:]
top30_idx_in_morgan150 = np.argsort(morgan_importances)[::-1][:30]
top50_idx_in_morgan150 = np.argsort(morgan_importances)[::-1][:50]

# ✅ 원본 Morgan2048 기준 인덱스 환산
top30_morgan_indices = morgan_top150_idx[top30_idx_in_morgan150]
top50_morgan_indices = morgan_top150_idx[top50_idx_in_morgan150]

# ✅ 저장
np.save('./_save/morgan_top30_from_combined.npy', top30_morgan_indices)
np.save('./_save/morgan_top50_from_combined.npy', top50_morgan_indices)

print("🎯 최종 저장 완료:")
print(" - Top30 index ➤ ./_save/morgan_top30_from_combined.npy")
print(" - Top50 index ➤ ./_save/morgan_top50_from_combined.npy")
