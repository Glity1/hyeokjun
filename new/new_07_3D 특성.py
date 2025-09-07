import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 경로 설정
save_dir = './_save/3d_descriptor_model/'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 로드
path = './_data/dacon/new/'
train_raw = pd.read_csv(path + 'train.csv')
test_raw = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ 2D + 3D descriptor 추출 함수
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 2D 특성
    feats = {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumRings': Descriptors.RingCount(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol)
    }

    # 3D 구조 생성
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol)

    # 3D 특성
    feats['Asphericity'] = rdMolDescriptors.CalcAsphericity(mol)
    feats['Eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
    feats['InertialShapeFactor'] = rdMolDescriptors.CalcInertialShapeFactor(mol)
    feats['NPR1'] = rdMolDescriptors.CalcNPR1(mol)
    feats['NPR2'] = rdMolDescriptors.CalcNPR2(mol)
    feats['RadiusOfGyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)
    feats['SpherocityIndex'] = rdMolDescriptors.CalcSpherocityIndex(mol)

    return feats

# ✅ 피처 생성
train_features = train_raw['Canonical_Smiles'].apply(featurize)
test_features = test_raw['Canonical_Smiles'].apply(featurize)

train_df = pd.DataFrame(train_features.tolist())
test_df = pd.DataFrame(test_features.tolist())
train_df['Inhibition'] = train_raw['Inhibition']

# ✅ 결측 제거
train_df.dropna(inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# ✅ 입력, 타겟 분리
X = train_df.drop(columns=['Inhibition'])
y = train_df['Inhibition']
X_test = test_df.copy()

# ✅ 스케일링
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ 모델 학습
model = RandomForestRegressor(random_state=222, n_estimators=300)
model.fit(X_scaled, y)

# ✅ 모델 저장
import joblib
model_path = os.path.join(save_dir, 'rf_3d_model.pkl')
joblib.dump(model, model_path)
print(f"💾 모델 저장 완료: {model_path}")

# ✅ 예측 및 평가
y_pred = model.predict(X_scaled)
rmse = mean_squared_error(y, y_pred, squared=False)
print(f"✅ RMSE: {rmse:.4f}")

# ✅ feature importance
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ✅ 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(30))
plt.title('📊 Top 30 Feature Importances (2D + 3D Descriptors)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_3d.png'))
plt.show()

# ✅ 제출 파일 생성
submit['Inhibition'] = model.predict(X_test_scaled)
submit.to_csv(os.path.join(path, 'rf_3d_submission.csv'), index=False)
print("📄 제출 파일 저장 완료: rf_3d_submission.csv")
