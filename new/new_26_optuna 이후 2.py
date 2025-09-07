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
import platform
import optuna
from optuna.samplers import TPESampler

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 로드
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values
smiles_train = train['Canonical_Smiles'].tolist()
smiles_test = test['Canonical_Smiles'].tolist()

# 3. 피처 추출
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
valid_train_idx = [i for i, f in enumerate(train_features) if f is not None]
train_features = [train_features[i] for i in valid_train_idx]
y = y[valid_train_idx]
valid_test_idx = [i for i, f in enumerate(test_features) if f is not None]
test_features = [test_features[i] for i in valid_test_idx]
X = np.array(train_features)
X_test = np.array(test_features)

# 4. 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. 중요도 기반 피처 선택 + 상관관계 제거
gb = GradientBoostingRegressor()
gb.fit(X_scaled, y)
importances = gb.feature_importances_
top_idx = np.argsort(importances)[::-1][:100]
X_top = X_scaled[:, top_idx]
X_test_top = X_test_scaled[:, top_idx]

df_top = pd.DataFrame(X_top)
corr_matrix = df_top.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
X_final = df_top.drop(df_top.columns[drop_cols], axis=1).values
X_test_final = pd.DataFrame(X_test_top).drop(drop_cols, axis=1).values

# 6. Train/Val split
x_train, x_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 7. 기본 모델 학습 (Pseudo Label 생성을 위한)
def build_dnn_default():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_final.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

dnn = build_dnn_default()
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
dnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=64, callbacks=[es], verbose=0)
xgb = XGBRegressor()
xgb.fit(x_train, y_train)

# 8. Confident Pseudo Label 생성
pseudo_dnn = dnn.predict(X_test_final).reshape(-1)
pseudo_xgb = xgb.predict(X_test_final)
pseudo_label = 0.7 * pseudo_dnn + 0.3 * pseudo_xgb
q1, q3 = np.percentile(pseudo_label, [25, 75])
conf_mask = (pseudo_label >= q1) & (pseudo_label <= q3)
X_conf = X_test_final[conf_mask]
y_conf = pseudo_label[conf_mask]

# 9. Pseudo 포함 데이터 재구성
X_pseudo = np.concatenate([X_final, X_conf], axis=0)
y_pseudo = np.concatenate([y, y_conf], axis=0)
x_train, x_val, y_train, y_val = train_test_split(X_pseudo, y_pseudo, test_size=0.2, random_state=42)

# 10. DNN Optuna 튜닝
def objective_dnn(trial):
    n_layer = trial.suggest_int('n_layer', 2, 5)
    units = trial.suggest_categorical('units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    for _ in range(n_layer - 1):
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100, batch_size=64, callbacks=[es], verbose=0)
    preds = model.predict(x_val).reshape(-1)
    return mean_squared_error(y_val, preds, squared=False)

study_dnn = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_dnn.optimize(objective_dnn, n_trials=30)
best_dnn_params = study_dnn.best_params
print("🎯 Best DNN Params:", best_dnn_params)

# 11. XGB Optuna 튜닝
def objective_xgb(trial):
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
    return mean_squared_error(y_val, preds, squared=False)

study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb_params = study_xgb.best_params
print("🎯 Best XGB Params:", best_xgb_params)

# 12. 최적 파라미터로 재학습
def build_best_dnn(params):
    model = Sequential()
    model.add(Dense(params['units'], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    for _ in range(params['n_layer'] - 1):
        model.add(Dense(params['units'], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

dnn = build_best_dnn(best_dnn_params)
dnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=64, callbacks=[es], verbose=0)

xgb = XGBRegressor(**best_xgb_params)
xgb.fit(x_train, y_train)

# 13. Voting 최적화
pred_dnn_val = dnn.predict(x_val).reshape(-1)
pred_xgb_val = xgb.predict(x_val)
best_rmse, best_w = float('inf'), 0.5
for w in np.linspace(0.1, 0.9, 9):
    blend = w * pred_dnn_val + (1 - w) * pred_xgb_val
    rmse = mean_squared_error(y_val, blend, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w
print(f"✅ 최종 Val RMSE (DNN {best_w:.2f} / XGB {1-best_w:.2f}): {best_rmse:.4f}")

# 14. 최종 예측 및 저장
pred_dnn_test = dnn.predict(X_test_final).reshape(-1)
pred_xgb_test = xgb.predict(X_test_final)
final_pred = best_w * pred_dnn_test + (1 - best_w) * pred_xgb_test
submit['Inhibition'] = final_pred
submit.to_csv('./_save/final_optuna_pseudo_voting.csv', index=False)
print("📁 최종 제출 파일 저장 완료!")
