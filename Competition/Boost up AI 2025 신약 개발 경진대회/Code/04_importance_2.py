import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Seed 고정
seed = 222
random.seed(seed)
np.random.seed(seed)

# ✅ 데이터 불러오기
path = './_data/dacon/new/'
train_raw = pd.read_csv(path + 'train.csv')
test_raw = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ RDKit 피처 추출
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
train_df = pd.DataFrame(train_features.tolist())
train_df['Inhibition'] = train_raw['Inhibition']

# ✅ 불필요 컬럼 제거
drop_cols = ['ExactMolWt', 'MolWt_TPSA', 'RotBond2']
train_df.drop(columns=drop_cols, inplace=True)

# ✅ log1p 변환
log_cols = ['MolWt', 'TPSA', 'HeavyAtomCount']
for col in log_cols:
    train_df[col] = np.log1p(train_df[col])

# ✅ 이진화
bin_cols = ['NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
for col in bin_cols:
    train_df[f'{col}_bin'] = (train_df[col] > 0).astype(int)

# ✅ 스케일링
X = train_df.drop(columns=['Inhibition'])
y = train_df['Inhibition']
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 모델 정의
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed, verbosity=0)

models = [model1, model2, model3, model4]
model_names = ['Decision', 'Random Forest', 'Gradient Boosting', 'XGBoost']

# ✅ 학습, 평가, 중요도 시각화
for model, name in zip(models, model_names):
    model.fit(X_scaled, y)
    pred = model.predict(X_scaled)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    importances = model.feature_importances_

    # 시각화
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=X.columns)
    plt.title(f'{name} Feature Importance (RMSE: {rmse:.4f})')
    plt.tight_layout()
    plt.show()
