# 1. 라이브러리
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import joblib

# 2. 데이터 로딩
path = './_data/dacon/new/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# 3. 피처 생성 함수
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
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol)
    ]
    features.extend(descriptors)
    return features

# 4. 피처 생성
train_features = train['Canonical_Smiles'].apply(featurize)
test_features = test['Canonical_Smiles'].apply(featurize)

# 5. 결측 제거
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

# 7. PCA 차원 축소
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 8. 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# 9. DNN 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_pca.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=500,
          batch_size=32,
          callbacks=[es],
          verbose=1)

# 10. 성능 평가
train_pred = model.predict(x_train).reshape(-1)
val_pred = model.predict(x_val).reshape(-1)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"✅ Train RMSE: {train_rmse:.4f}")
print(f"✅ Val RMSE  : {val_rmse:.4f}")

# # 11. 테스트 예측 및 제출 저장
# test_pred = model.predict(X_test_pca).reshape(-1)
# submit['Inhibition'] = test_pred
# submit_path = './_save/dnn_pca100_submission.csv'
# submit.to_csv(submit_path, index=False)
# print(f"📁 제출 파일 저장 완료: {submit_path}")

# # 12. 저장
# model.save('./_save/dnn_pca100_model.h5')
# joblib.dump(scaler, './_save/dnn_pca100_scaler.pkl')
# joblib.dump(pca, './_save/dnn_pca100_pca.pkl')
