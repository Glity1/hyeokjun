# ✅ ±0.8σ 기준 Pseudo Label 강화 DNN 전체 코드

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# 1. 전처리 도구 로딩
scaler = joblib.load('./_save/dnn_optuna_scaler.pkl')
selector = joblib.load('./_save/dnn_optuna_selector.pkl')
model = load_model('./_save/dnn_optuna_model.h5')

# 2. 데이터 불러오기
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# 3. Featurize 함수 정의
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

# 4. Test 데이터 featurize 및 변환
test_features = test['Canonical_Smiles'].apply(featurize)
test_features = test_features[test_features.notnull()]
X_test = np.array(test_features.tolist())
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# 5. 모델 예측 및 ±0.8σ 기준 confident 추출
test_pred = model.predict(X_test_selected).reshape(-1)
mean = np.mean(test_pred)
std = np.std(test_pred)
lower = mean - 0.8 * std
upper = mean + 0.8 * std
confident_idx = np.where((test_pred >= lower) & (test_pred <= upper))[0]
confident_X = X_test_selected[confident_idx]
confident_y = test_pred[confident_idx]

# 6. Train 데이터 불러오기 및 featurize
train = pd.read_csv('./_data/dacon/new/train.csv')
train_features = train['Canonical_Smiles'].apply(featurize)
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
X_train = np.array(train_features.tolist())
y_train = train['Inhibition'].values
X_train_scaled = scaler.transform(X_train)
X_train_selected = selector.transform(X_train_scaled)

# 7. Train + Confident Test 합치기
X_aug = np.concatenate([X_train_selected, confident_X], axis=0)
y_aug = np.concatenate([y_train, confident_y], axis=0)
x_train, x_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

# 8. Optuna 기반 DNN 구조로 학습
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

# 9. 평가 및 제출
train_rmse = np.sqrt(mean_squared_error(y_train, model_new.predict(x_train).reshape(-1)))
val_rmse = np.sqrt(mean_squared_error(y_val, model_new.predict(x_val).reshape(-1)))
print(f"✅ (Pseudo ±0.8σ) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Pseudo ±0.8σ) Val   RMSE: {val_rmse:.4f}")

final_pred = model_new.predict(X_test_selected).reshape(-1)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/dnn_optuna_confident_08sigma_submission.csv', index=False)
model_new.save('./_save/dnn_optuna_confident_08sigma_model.h5')
