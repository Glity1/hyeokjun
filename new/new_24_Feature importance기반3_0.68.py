# 1. 라이브러리
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
import seaborn as sns

# 2. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values
smiles_train = train['Canonical_Smiles'].tolist()
smiles_test = test['Canonical_Smiles'].tolist()

# 3. 피처 추출 함수
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

# 4. 피처 생성 (Train/Test)
train_features = [featurize(s) for s in smiles_train]
test_features = [featurize(s) for s in smiles_test]

# 결측치 제거
valid_train_idx = [i for i, f in enumerate(train_features) if f is not None]
train_features = [train_features[i] for i in valid_train_idx]
y = y[valid_train_idx]

valid_test_idx = [i for i, f in enumerate(test_features) if f is not None]
test_features = [test_features[i] for i in valid_test_idx]

X = np.array(train_features)
X_test = np.array(test_features)

# 5. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. GradientBoosting으로 중요도 계산
gb = GradientBoostingRegressor()
gb.fit(X_scaled, y)
importances = gb.feature_importances_

# 상위 100개 피처
top_idx = np.argsort(importances)[::-1][:100]
X_top = X_scaled[:, top_idx]
X_test_top = X_test_scaled[:, top_idx]

# 7. 상관관계 기반 Drop
df_top = pd.DataFrame(X_top)
corr_matrix = df_top.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]

# drop 적용
X_final = df_top.drop(df_top.columns[drop_cols], axis=1).values
X_test_final = pd.DataFrame(X_test_top).drop(drop_cols, axis=1).values

# 8. Train/Val Split
x_train, x_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 9. DNN 학습
dnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_final.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
dnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = dnn.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# 10. XGBoost 학습
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(x_train, y_train)

# 11. Pseudo Label 생성
pseudo_dnn = dnn.predict(X_test_final).reshape(-1)
pseudo_xgb = xgb.predict(X_test_final)
pseudo_label = (0.7 * pseudo_dnn + 0.3 * pseudo_xgb)

# 12. 훈련 데이터에 pseudo 추가
X_pseudo = np.concatenate([X_final, X_test_final], axis=0)
y_pseudo = np.concatenate([y, pseudo_label], axis=0)

# 13. 재학습: DNN + XGB
x_train, x_val, y_train, y_val = train_test_split(X_pseudo, y_pseudo, test_size=0.2, random_state=42)

# DNN 재학습
dnn.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=300, batch_size=64, callbacks=[es], verbose=1)

# XGB 재학습
xgb.fit(x_train, y_train)

# 14. 앙상블 예측
pred_dnn_val = dnn.predict(x_val).reshape(-1)
pred_xgb_val = xgb.predict(x_val)
val_pred = 0.7 * pred_dnn_val + 0.3 * pred_xgb_val

val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"✅ (Pseudo Ensemble) Val RMSE: {val_rmse:.4f}")

# 15. 최종 제출
pred_dnn_final = dnn.predict(X_test_final).reshape(-1)
pred_xgb_final = xgb.predict(X_test_final)
final_pred = 0.7 * pred_dnn_final + 0.3 * pred_xgb_final

submit['Inhibition'] = final_pred
submit.to_csv('./_save/final_pseudo_ensemble_submission.csv', index=False)

# 1. 라이브러리
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
...

# ✅ 한글 깨짐 방지 설정
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 16. 시각화 (Loss Curve)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('DNN 학습 곡선')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/final_loss_curve.png')
plt.show()

# 17. 시각화 (예측 vs 실제)
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('검증 데이터: 예측 vs 실제')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/final_prediction_vs_actual.png')
plt.show()
