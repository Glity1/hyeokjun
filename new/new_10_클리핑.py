# 1. 라이브러리
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

# 2. 데이터 불러오기
path = './_data/dacon/new/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# 3. 피처 추출 함수: Fingerprint + RDKit Descriptors
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    features = list(fp)
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.HeavyAtomCount(mol)
    ]
    features.extend(descriptors)
    return features

# 4. 피처 생성
train_features = train['Canonical_Smiles'].apply(featurize)
test_features = test['Canonical_Smiles'].apply(featurize)

train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
test_features = test_features[test_features.notnull()]

X = np.array(train_features.tolist())
X_test = np.array(test_features.tolist())
y = train['Inhibition'].values

# 5. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y)

# 7. 예측
y_pred = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Train RMSE: {rmse:.4f}")

# 8. 테스트 예측 및 제출 저장
test_pred = model.predict(X_test_scaled)
submit['Inhibition'] = test_pred
submit_path = './_save/rf_hybrid_submission.csv'
submit.to_csv(submit_path, index=False)
print(f"제출 파일 저장 완료: {submit_path}")

# 9. 모델 저장
joblib.dump(model, './_save/rf_hybrid_model.pkl')
joblib.dump(scaler, './_save/rf_hybrid_scaler.pkl')
