import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

# Seed 고정
seed = 222
random.seed(seed)
np.random.seed(seed)

# ✅ 경로 설정
path = './_data/dacon/new/'
save_dir = './_save/xgb_no_bin/'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 불러오기
train_raw = pd.read_csv(path + 'train.csv')
test_raw = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ RDKit 피처 추출 함수
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumRings': Descriptors.RingCount(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'MolWt_TPSA': Descriptors.MolWt(mol) * Descriptors.TPSA(mol),
        'Donor/Acceptor': Descriptors.NumHDonors(mol) / (Descriptors.NumHAcceptors(mol) + 1),
        'RotBond2': Descriptors.NumRotatableBonds(mol) ** 2
    }

# ✅ 피처 생성
train_features = train_raw['Canonical_Smiles'].apply(featurize)
test_features = test_raw['Canonical_Smiles'].apply(featurize)
train_df = pd.DataFrame(train_features.tolist())
test_df = pd.DataFrame(test_features.tolist())
train_df['Inhibition'] = train_raw['Inhibition']

# ✅ 불필요 컬럼 제거
drop_cols = ['ExactMolWt', 'MolWt_TPSA', 'RotBond2']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# ✅ log1p 변환
log_cols = ['MolWt', 'TPSA', 'HeavyAtomCount']
for col in log_cols:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])

# ✅ 이진화 피처 생성 후 제거
bin_cols = ['NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
for col in bin_cols:
    train_df[f'{col}_bin'] = (train_df[col] > 0).astype(int)
    test_df[f'{col}_bin'] = (test_df[col] > 0).astype(int)

# ✅ 중요도 낮은 _bin 피처 제거
remove_bin_cols = [f'{col}_bin' for col in bin_cols]
train_df.drop(columns=remove_bin_cols, inplace=True)
test_df.drop(columns=remove_bin_cols, inplace=True)

# ✅ 스케일링
X = train_df.drop(columns=['Inhibition'])
y = train_df['Inhibition']
X_test = test_df.copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ XGBoost 모델 학습
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=seed,
    verbosity=0
)
model.fit(X_scaled, y)

# ✅ 모델 및 가중치 저장
model_path = os.path.join(save_dir, 'xgb_model_no_bin.model')
model.save_model(model_path)
print(f"💾 모델 구조 저장 완료: {model_path}")

pkl_path = os.path.join(save_dir, 'xgb_model_no_bin.pkl')
joblib.dump(model, pkl_path)
print(f"💾 모델 가중치 저장 완료: {pkl_path}")

# ✅ 예측 및 평가
y_pred = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"✅ XGBoost RMSE (after dropping _bin features): {rmse:.4f}")

# ✅ Feature Importance 시각화
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title(f"XGBoost Feature Importance (RMSE: {rmse:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xgb_feature_importance_no_bin.png'))
plt.show()

from scipy.stats import pearsonr
corr, _ = pearsonr(y, y_pred)
print(f"📈 Pearson correlation (B): {corr:.4f}")

sns.kdeplot(y, label='True')
sns.kdeplot(y_pred, label='Pred')
plt.legend()
plt.title("True vs Predicted Distribution")
plt.show()

# ✅ 제출 저장
submit['Inhibition'] = model.predict(X_test_scaled)
submit_path = os.path.join(path, 'xgb_submission_no_bin.csv')
submit.to_csv(submit_path, index=False)
print(f"📄 제출 파일 저장 완료: {submit_path}")
