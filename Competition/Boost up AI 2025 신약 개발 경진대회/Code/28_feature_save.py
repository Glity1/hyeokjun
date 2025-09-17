import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
y = train['Inhibition'].values
smiles = train['Canonical_Smiles'].tolist()

# 2. 피처 추출 함수
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    features = list(fp)
    descriptors = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol), Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol), Descriptors.RingCount(mol),
        Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol), Descriptors.ExactMolWt(mol),
        Descriptors.MolMR(mol), Descriptors.LabuteASA(mol),
        Descriptors.BalabanJ(mol), Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol)
    ]
    features.extend(descriptors)
    return features

# 3. 전체 피처 추출
features = [featurize(smi) for smi in smiles]
valid_idx = [i for i, f in enumerate(features) if f is not None]
X = np.array([features[i] for i in valid_idx])
y = y[valid_idx]

# 4. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. GradientBoosting 기반 중요도 추출
gb = GradientBoostingRegressor()
gb.fit(X_scaled, y)
importances = gb.feature_importances_

# 상위 100개 피처 선택
top_idx = np.argsort(importances)[::-1][:100]
X_top = X_scaled[:, top_idx]

# 6. 상관관계 필터링
df_top = pd.DataFrame(X_top)
corr_matrix = df_top.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_filtered = df_top.drop(df_top.columns[to_drop], axis=1).values

# 7. PCA (설명 분산 95%)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_filtered)

# 8. DNN 학습용 분할
x_train, x_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 9. 결과 저장
os.makedirs('./_save/', exist_ok=True)
np.save('./_save/final_x_train.npy', x_train)
np.save('./_save/final_x_val.npy', x_val)
np.save('./_save/final_y_train.npy', y_train)
np.save('./_save/final_y_val.npy', y_val)
joblib.dump(scaler, './_save/final_scaler.pkl')
joblib.dump(pca, './_save/final_pca.pkl')
joblib.dump(top_idx, './_save/final_top_idx.pkl')
joblib.dump(to_drop, './_save/final_drop_cols.pkl')
