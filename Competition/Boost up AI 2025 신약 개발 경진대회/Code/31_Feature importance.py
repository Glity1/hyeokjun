# 1. 라이브러리
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# 2. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
y = train['Inhibition'].values
smiles = train['Canonical_Smiles'].tolist()

# 3. 피처 추출
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
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

features = [featurize(smi) for smi in smiles]
valid_idx = [i for i, f in enumerate(features) if f is not None]
X = np.array([features[i] for i in valid_idx])
y = y[valid_idx]

# 4. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. Feature Importance 기반 상위 k개 선택
k = 64  # 사용할 특성 개수
gbr = GradientBoostingRegressor()
gbr.fit(X_scaled, y)
importances = gbr.feature_importances_
top_idx = np.argsort(importances)[::-1][:k]
X_top = X_scaled[:, top_idx]

# 6. 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(X_top, y, test_size=0.2, random_state=42)

# 7. DNN 모델
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_top.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=300,
          batch_size=64,
          callbacks=[es],
          verbose=1)

# 8. 평가
train_rmse = mean_squared_error(y_train, model.predict(x_train).reshape(-1), squared=False)
val_rmse = mean_squared_error(y_val, model.predict(x_val).reshape(-1), squared=False)

print(f"✅ (FI-Top{len(top_idx)}) Train RMSE: {train_rmse:.4f}")
print(f"✅ (FI-Top{len(top_idx)}) Val   RMSE: {val_rmse:.4f}")

# 9. 저장
os.makedirs('./_save/', exist_ok=True)
np.save('./_save/fi_x_train.npy', x_train)
np.save('./_save/fi_x_val.npy', x_val)
np.save('./_save/fi_y_train.npy', y_train)
np.save('./_save/fi_y_val.npy', y_val)
model.save('./_save/fi_dnn_model.h5')
joblib.dump(scaler, './_save/fi_scaler.pkl')
joblib.dump(top_idx, './_save/fi_top_idx.pkl')
print("📁 모델 및 데이터 저장 완료")
