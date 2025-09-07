# ✅ 기존 라이브러리들은 그대로 유지
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
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ✅ 데이터 로드 및 전처리 동일
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values
smiles_train = train['Canonical_Smiles'].tolist()
smiles_test = test['Canonical_Smiles'].tolist()

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

train_features = [featurize(s) for s in smiles_train]
test_features = [featurize(s) for s in smiles_test]
valid_train_idx = [i for i, f in enumerate(train_features) if f is not None]
train_features = [train_features[i] for i in valid_train_idx]
y = y[valid_train_idx]
valid_test_idx = [i for i, f in enumerate(test_features) if f is not None]
test_features = [test_features[i] for i in valid_test_idx]
X = np.array(train_features)
X_test = np.array(test_features)

# ✅ Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ Feature Importance + Correlation 제거
gb = GradientBoostingRegressor()
gb.fit(X_scaled, y)
top_idx = np.argsort(gb.feature_importances_)[::-1][:100]
X_top = X_scaled[:, top_idx]
X_test_top = X_test_scaled[:, top_idx]
df_top = pd.DataFrame(X_top)
corr_matrix = df_top.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
X_final = df_top.drop(df_top.columns[drop_cols], axis=1).values
X_test_final = pd.DataFrame(X_test_top).drop(drop_cols, axis=1).values

# ✅ Train/Validation split
x_train, x_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ✅ Optuna DNN 최적화
def objective(trial):
    n_layer = trial.suggest_int('n_layer', 2, 4)
    units = trial.suggest_categorical('units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(n_layer - 1):
        model.add(Dense(units // 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100, batch_size=64, callbacks=[es], verbose=0)

    pred = model.predict(x_val).reshape(-1)
    rmse = mean_squared_error(y_val, pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)
best_trial = study.best_trial
print("✅ DNN Optuna Best Params:", best_trial.params)

# ✅ 최적 파라미터로 DNN 재학습
def build_best_dnn(params):
    model = Sequential()
    model.add(Dense(params['units'], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    for _ in range(params['n_layer'] - 1):
        model.add(Dense(params['units'] // 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

dnn = build_best_dnn(best_trial.params)
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = dnn.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=300, batch_size=64, callbacks=[es], verbose=1)


# 8. Optuna를 통한 XGBoost 튜닝
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=20, verbose=False)
    preds = model.predict(x_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("🎯 Best XGB Params:", best_params)
xgb = XGBRegressor(**best_params)
xgb.fit(x_train, y_train)

# 9. Confident Pseudo Label
pseudo_dnn = dnn.predict(X_test_final).reshape(-1)
pseudo_xgb = xgb.predict(X_test_final)
pseudo_label = 0.7 * pseudo_dnn + 0.3 * pseudo_xgb
lower, upper = np.percentile(pseudo_label, [25, 75])
conf_mask = (pseudo_label >= lower) & (pseudo_label <= upper)
X_conf = X_test_final[conf_mask]
y_conf = pseudo_label[conf_mask]

# 10. Pseudo 포함 재학습
X_pseudo = np.concatenate([X_final, X_conf], axis=0)
y_pseudo = np.concatenate([y, y_conf], axis=0)
x_train, x_val, y_train, y_val = train_test_split(X_pseudo, y_pseudo, test_size=0.2, random_state=42)
dnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=64, callbacks=[es], verbose=1)
xgb.fit(x_train, y_train)

# 11. 앙상블 예측
pred_dnn_val = dnn.predict(x_val).reshape(-1)
pred_xgb_val = xgb.predict(x_val)
val_pred = 0.7 * pred_dnn_val + 0.3 * pred_xgb_val
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"✅ (Optuna + Confident Pseudo) Val RMSE: {val_rmse:.4f}")

# DNN 재학습
dnn.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=300, batch_size=64, callbacks=[es], verbose=1)

# XGB 재학습
xgb.fit(x_train, y_train)

# ✅ validation 예측
pred_dnn_val = dnn.predict(x_val).reshape(-1)
pred_xgb_val = xgb.predict(x_val)

# ✅ 여기 아래에 붙이면 됨! (Voting 최적화)
from sklearn.metrics import mean_squared_error
import numpy as np

best_rmse = float('inf')
best_w = 0.5
rmse_list = []

for w in np.linspace(0.1, 0.9, 9):
    blended = w * pred_dnn_val + (1 - w) * pred_xgb_val
    rmse = mean_squared_error(y_val, blended, squared=False)
    rmse_list.append((w, rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print("🔍 Voting 최적화 결과:")
for w, rmse in rmse_list:
    print(f"  DNN: {w:.2f}, XGB: {1-w:.2f}, Val RMSE: {rmse:.4f}")
print(f"\n✅ 최적 가중치 → DNN: {best_w:.2f}, XGB: {1-best_w:.2f}, 최소 RMSE: {best_rmse:.4f}")

# ✅ 최종 예측
pred_dnn_final = dnn.predict(X_test_final).reshape(-1)
pred_xgb_final = xgb.predict(X_test_final)
final_pred = best_w * pred_dnn_final + (1 - best_w) * pred_xgb_final

submit['Inhibition'] = final_pred
submit.to_csv('./_save/final_voting_optimized_submission.csv', index=False)
print("📁 최종 제출 파일 저장 완료!")

# 13. 시각화
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
