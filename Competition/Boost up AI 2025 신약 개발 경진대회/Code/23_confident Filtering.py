# 1. 기본 설정 및 전처리기/모델 로딩
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 전처리기 및 모델 로드
scaler = joblib.load('./_save/dnn_optuna_scaler.pkl')
selector = joblib.load('./_save/dnn_optuna_selector.pkl')
model = load_model('./_save/dnn_optuna_model.h5')

# 테스트 데이터
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# 2. Featurize 함수
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

# 3. Test 데이터 처리 및 ±0.8σ confident 예측 샘플 추출
test_features = test['Canonical_Smiles'].apply(featurize)
test_features = test_features[test_features.notnull()]
X_test = np.array(test_features.tolist())

X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

test_pred = model.predict(X_test_selected).reshape(-1)
mean = np.mean(test_pred)
std = np.std(test_pred)

lower, upper = mean - 0.8 * std, mean + 0.8 * std
confident_idx = np.where((test_pred >= lower) & (test_pred <= upper))[0]
confident_X = X_test_selected[confident_idx]
confident_y = test_pred[confident_idx]

print(f"✅ (±0.8σ) Confident Test 샘플 수: {len(confident_y)}")

# 4. 기존 Train 데이터 처리
train = pd.read_csv('./_data/dacon/new/train.csv')
train_features = train['Canonical_Smiles'].apply(featurize)
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
X_train = np.array(train_features.tolist())
y_train = train['Inhibition'].values

X_train_scaled = scaler.transform(X_train)
X_train_selected = selector.transform(X_train_scaled)

# 5. Soft Label + Sample Weight 구성
X_aug = np.concatenate([X_train_selected, confident_X], axis=0)
y_aug = np.concatenate([y_train, confident_y], axis=0)

weights_train = np.ones(len(y_train))
conf_dist = np.abs(confident_y - mean)
weights_conf = 1 / (1 + conf_dist)
sample_weights = np.concatenate([weights_train, weights_conf], axis=0)

# 6. DNN 재학습
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
    X_aug, y_aug, sample_weights, test_size=0.2, random_state=42
)

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
              sample_weight=w_train,
              epochs=300,
              batch_size=64,
              callbacks=[es],
              verbose=1)

# 7. 평가 및 제출
train_rmse = np.sqrt(mean_squared_error(y_train, model_new.predict(x_train).reshape(-1)))
val_rmse = np.sqrt(mean_squared_error(y_val, model_new.predict(x_val).reshape(-1)))

print(f"✅ (Pseudo ±0.8σ) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Pseudo ±0.8σ) Val   RMSE: {val_rmse:.4f}")

final_pred = model_new.predict(X_test_selected).reshape(-1)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/dnn_softlabel_08sigma_submission.csv', index=False)
model_new.save('./_save/dnn_softlabel_08sigma_model.h5')
print("📁 ±0.8σ Soft Label + SampleWeight 결과 저장 완료")
