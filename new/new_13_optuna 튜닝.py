# 1. 라이브러리
import pandas as pd
import numpy as np
import optuna
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 2. 데이터 로딩
path = './_data/dacon/new/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 3. 피처 생성 함수
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

# 4. 피처 생성
train_features = train['Canonical_Smiles'].apply(featurize)
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
X = np.array(train_features.tolist())
y = train['Inhibition'].values

# 5. 스케일링 및 Top 50 선택
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

xgb = XGBRegressor(random_state=42)
xgb.fit(X_scaled, y)
selector = SelectFromModel(xgb, threshold=-np.inf, max_features=50, prefit=True)
X_selected = selector.transform(X_scaled)

# 6. 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# 7. Optuna 목적 함수
def objective(trial):
    model = Sequential()
    n_layers = trial.suggest_int("n_layers", 1, 3)

    for i in range(n_layers):
        units = trial.suggest_int(f"units_{i}", 32, 256)
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        dropout_rate = trial.suggest_float(f"dropout_{i}", 0.0, 0.5)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # 출력층

    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=100,
              batch_size=batch_size,
              callbacks=[es],
              verbose=0)

    val_pred = model.predict(x_val).reshape(-1)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return val_rmse

# 8. Optuna 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# 9. 결과 출력
print("🎯 Best Trial:")
print(study.best_trial)
print("🧪 Best Params:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")
