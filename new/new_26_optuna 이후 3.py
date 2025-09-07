# ✅ 1. 라이브러리
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import platform

# ✅ 2. 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ✅ 3. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values
smiles_train = train['Canonical_Smiles'].tolist()
smiles_test = test['Canonical_Smiles'].tolist()

# ✅ 4. RDKit 특성 추출 함수
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
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

train_features = [featurize(s) for s in smiles_train]
test_features = [featurize(s) for s in smiles_test]
train = pd.DataFrame([f for f in train_features if f is not None])
test = pd.DataFrame([f for f in test_features if f is not None])
y = y[[i for i, f in enumerate(train_features) if f is not None]]

# ✅ 5. Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train)
X_test_scaled = scaler.transform(test)

# ✅ 6. Feature Importance 기반 상위 100개 선택
gb = GradientBoostingRegressor()
gb.fit(X_scaled, y)
importances = gb.feature_importances_
top_idx = np.argsort(importances)[::-1][:100]
X_top = X_scaled[:, top_idx]
X_test_top = X_test_scaled[:, top_idx]

# ✅ 7. 상관관계 기반 중복 제거
corr_matrix = pd.DataFrame(X_top).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
X_final = pd.DataFrame(X_top).drop(drop_cols, axis=1).values
X_test_final = pd.DataFrame(X_test_top).drop(drop_cols, axis=1).values

# ✅ 8. Train/Val Split
x_train, x_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ✅ 9. DNN 모델 구성
best_dnn_params = {'n_layer': 4, 'units': 128, 'dropout_rate': 0.5758, 'learning_rate': 0.00616}
def build_dnn():
    model = Sequential()
    model.add(Dense(best_dnn_params['units'], activation='relu', input_shape=(X_final.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(best_dnn_params['dropout_rate']))
    for _ in range(best_dnn_params['n_layer'] - 1):
        model.add(Dense(best_dnn_params['units'], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(best_dnn_params['dropout_rate']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_dnn_params['learning_rate']), loss='mse')
    return model

dnn = build_dnn()
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
dnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=64, callbacks=[es], verbose=0)

# ✅ 10. XGB 모델 구성 및 학습
best_xgb_params = {
    'n_estimators': 510,
    'max_depth': 6,
    'learning_rate': 0.0131,
    'subsample': 0.505,
    'colsample_bytree': 0.766,
    'reg_alpha': 2.193,
    'reg_lambda': 0.0027,
    'random_state': 42
}
xgb = XGBRegressor(**best_xgb_params)
xgb.fit(x_train, y_train)

# ✅ 11. 앙상블 예측 → pseudo 생성 기준
pred_dnn_test = dnn.predict(X_test_final).reshape(-1)
pred_xgb_test = xgb.predict(X_test_final)
pseudo_pred = 0.5 * pred_dnn_test + 0.5 * pred_xgb_test

lower, upper = np.percentile(pseudo_pred, [25, 75])
conf_mask = (pseudo_pred >= lower) & (pseudo_pred <= upper)
X_conf = X_test_final[conf_mask]
y_conf = pseudo_pred[conf_mask]

# ✅ 12. Pseudo 포함 학습
X_pseudo = np.concatenate([X_final, X_conf], axis=0)
y_pseudo = np.concatenate([y, y_conf], axis=0)
x_train, x_val, y_train, y_val = train_test_split(X_pseudo, y_pseudo, test_size=0.2, random_state=42)

dnn = build_dnn()
dnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=64, callbacks=[es], verbose=0)
xgb.fit(x_train, y_train)

# ✅ 13. Validation 앙상블
val_dnn = dnn.predict(x_val).reshape(-1)
val_xgb = xgb.predict(x_val)
val_final = 0.5 * val_dnn + 0.5 * val_xgb
val_rmse = mean_squared_error(y_val, val_final, squared=False)
print(f"\n✅ 최종 Val RMSE (DNN 0.5 / XGB 0.5): {val_rmse:.4f}")

# ✅ 14. 최종 제출
final_dnn = dnn.predict(X_test_final).reshape(-1)
final_xgb = xgb.predict(X_test_final)
final_pred = 0.5 * final_dnn + 0.5 * final_xgb
submit['Inhibition'] = final_pred
submit.to_csv('./_save/final_pseudo_by_ensemble2.csv', index=False)
print("\n📁 최종 제출 파일 저장 완료!")
