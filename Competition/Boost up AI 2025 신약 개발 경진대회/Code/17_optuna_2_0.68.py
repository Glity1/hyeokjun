# 1. 기존 전처리 및 모델 로딩 (전 단계 코드 그대로 사용)
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 로드
scaler = joblib.load('./_save/dnn_optuna_scaler.pkl')
selector = joblib.load('./_save/dnn_optuna_selector.pkl')
model = load_model('./_save/dnn_optuna_model.h5')

# 데이터
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# featurize 함수는 이전 코드와 동일하게 사용
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
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

# 2. Test 데이터 featurize 및 변환
test_features = test['Canonical_Smiles'].apply(featurize)
test_features = test_features[test_features.notnull()]
X_test = np.array(test_features.tolist())

X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# 3. 기존 모델로 Test 예측 → Confident 샘플 추출
test_pred = model.predict(X_test_selected).reshape(-1)

# 방법: 중앙값 기준 ±1σ 내 샘플 선택 (IQR 방식도 가능)
mean = np.mean(test_pred)
std = np.std(test_pred)

lower = mean - std
upper = mean + std

confident_idx = np.where((test_pred >= lower) & (test_pred <= upper))[0]
confident_X = X_test_selected[confident_idx]
confident_y = test_pred[confident_idx]

print(f"✅ Confident Test 샘플 수: {len(confident_y)}")

# 4. 기존 학습 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
train_features = train['Canonical_Smiles'].apply(featurize)
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
X_train = np.array(train_features.tolist())
y_train = train['Inhibition'].values

X_train_scaled = scaler.transform(X_train)
X_train_selected = selector.transform(X_train_scaled)

# 5. 원래 train + confident test → 확장 학습 데이터 구성
X_aug = np.concatenate([X_train_selected, confident_X], axis=0)
y_aug = np.concatenate([y_train, confident_y], axis=0)

# 6. 데이터 분할 후 DNN 재학습
x_train, x_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Optuna 최적 구조 그대로 재사용
model_new = Sequential([
    Dense(42, activation='relu', input_shape=(X_aug.shape[1],)),
    BatchNormalization(),
    Dropout(0.3717),
    Dense(79, activation='relu'),
    BatchNormalization(),
    Dropout(0.2992),
    Dense(1)
])

model_new.compile(optimizer=Adam(learning_rate=0.00984), loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_new.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=300,
              batch_size=64,
              callbacks=[es],
              verbose=1)

# 7. 평가 및 제출
train_rmse = np.sqrt(mean_squared_error(y_train, model_new.predict(x_train).reshape(-1)))
val_rmse = np.sqrt(mean_squared_error(y_val, model_new.predict(x_val).reshape(-1)))

print(f"✅ (Pseudo) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Pseudo) Val   RMSE: {val_rmse:.4f}")

final_pred = model_new.predict(X_test_selected).reshape(-1)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/dnn_optuna_confident_submission.csv', index=False)
print("📁 최종 제출 파일 저장 완료")
model_new.save('./_save/dnn_optuna_confident_model.h5')
