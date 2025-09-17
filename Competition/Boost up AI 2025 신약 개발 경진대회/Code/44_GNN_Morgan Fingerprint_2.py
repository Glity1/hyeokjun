import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBRegressor, plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ✅ 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
smiles_list = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values

# ✅ Morgan Fingerprint 추출 함수 (2048 bits)
def get_morgan_fp(smiles, radius=2, n_bits=2048):
    morgan_features = []
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            morgan_features.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        morgan_features.append(arr)
    return morgan_features

# ✅ Morgan Feature 추출
morgan_features = get_morgan_fp(smiles_list)
X = np.array([m for m in morgan_features if m is not None])
y_filtered = np.array([yy for m, yy in zip(morgan_features, y) if m is not None])

# ✅ 데이터 분할 (검증용)
X_train, X_val, y_train, y_val = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

# ✅ XGBoost 모델 학습
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=False)

# ✅ 예측 및 성능 확인
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"✅ Validation RMSE: {rmse:.4f}")

# ✅ 특성 중요도 시각화
plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=30, importance_type='gain', height=0.5)
plt.title("Top 30 Morgan Feature Importances (by Gain)")
plt.tight_layout()
plt.show()

# ✅ 중요도 내림차순 정렬된 인덱스
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("Top 10 Important Morgan Indices:", sorted_idx[:10])
