# 1. 라이브러리
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import joblib

# 2. 데이터 불러오기
path = './_data/dacon/new/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# 3. 피처 추출 함수: Fingerprint + Descriptor
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

# 5. 결측치 제거
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
test_features = test_features[test_features.notnull()]

X = np.array(train_features.tolist())
X_test = np.array(test_features.tolist())
y = train['Inhibition'].values

# 6. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 7. Top 20 특성 선택
xgb = XGBRegressor(random_state=42)
xgb.fit(X_scaled, y)

selector = SelectFromModel(xgb, threshold=-np.inf, max_features=20, prefit=True)
X_selected = selector.transform(X_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 8. 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# 9. DNN 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_selected.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=300,
          batch_size=32,
          callbacks=[es],
          verbose=1)

# 10. RMSE 계산
train_pred = model.predict(x_train).reshape(-1)
val_pred = model.predict(x_val).reshape(-1)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"✅ Train RMSE: {train_rmse:.4f}")
print(f"✅ Val RMSE  : {val_rmse:.4f}")

# 11. 테스트 예측 및 제출 저장
y_pred_test = model.predict(X_test_selected).reshape(-1)
submit['Inhibition'] = y_pred_test
submit_path = './_save/dnn_top20_submission.csv'
submit.to_csv(submit_path, index=False)
print(f"📁 제출 파일 저장 완료: {submit_path}")

# 12. 모델 및 객체 저장
model.save('./_save/dnn_top20_model.h5')
joblib.dump(scaler, './_save/dnn_top20_scaler.pkl')
joblib.dump(selector, './_save/dnn_top20_selector.pkl')
