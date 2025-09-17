# 1. 라이브러리
import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer, AutoModel
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 2. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values

# 3. ChemBERTa 임베딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model_chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)

def get_chemberta_embedding(smiles_list):
    embeddings = []
    for smi in smiles_list:
        with torch.no_grad():
            inputs = tokenizer(smi, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model_chemberta(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
    return np.array(embeddings)

train_chemberta = get_chemberta_embedding(train['Canonical_Smiles'].tolist())
test_chemberta = get_chemberta_embedding(test['Canonical_Smiles'].tolist())

# 4. RDKit + Morgan
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

train_features = train['Canonical_Smiles'].apply(featurize)
test_features = test['Canonical_Smiles'].apply(featurize)

valid_idx = train_features.notnull()
train_features = train_features[valid_idx]
train_chemberta = train_chemberta[valid_idx]
y = y[valid_idx]

X_rdkit = np.array(train_features.tolist())
X_test_rdkit = np.array(test_features[test_features.notnull()].tolist())

# 5. 결합 및 전처리
X_all = np.concatenate([X_rdkit, train_chemberta], axis=1)
X_test_all = np.concatenate([X_test_rdkit, test_chemberta], axis=1)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)
X_test_scaled = scaler.transform(X_test_all)

selector = SelectKBest(score_func=f_regression, k=256)
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

joblib.dump(scaler, './_save/hybrid_scaler.pkl')
joblib.dump(selector, './_save/hybrid_selector.pkl')

# 6. Optuna 목적 함수
def objective(trial):
    x_train, x_val, y_train_, y_val_ = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(trial.suggest_int('units1', 64, 512), activation='relu', input_shape=(X_selected.shape[1],)))
    if trial.suggest_categorical('bn1', [True, False]):
        model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout1', 0.1, 0.5)))

    model.add(Dense(trial.suggest_int('units2', 32, 256), activation='relu'))
    if trial.suggest_categorical('bn2', [True, False]):
        model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout2', 0.1, 0.5)))

    model.add(Dense(1))

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(x_train, y_train_, validation_data=(x_val, y_val_),
              epochs=200, batch_size=64, callbacks=[es], verbose=0)

    pred = model.predict(x_val).reshape(-1)
    rmse = mean_squared_error(y_val_, pred, squared=False)
    return rmse

# 7. Optuna 실행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("✅ Best trial:", study.best_trial.params)

# 8. 최적 모델로 학습 후 저장
best_params = study.best_trial.params

model = Sequential()
model.add(Dense(best_params['units1'], activation='relu', input_shape=(X_selected.shape[1],)))
if best_params['bn1']: model.add(BatchNormalization())
model.add(Dropout(best_params['dropout1']))
model.add(Dense(best_params['units2'], activation='relu'))
if best_params['bn2']: model.add(BatchNormalization())
model.add(Dropout(best_params['dropout2']))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

x_train, x_val, y_train_, y_val_ = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train_, validation_data=(x_val, y_val_),
          epochs=300, batch_size=64, callbacks=[es], verbose=1)

# 9. 평가 및 저장
train_rmse = mean_squared_error(y_train_, model.predict(x_train).reshape(-1), squared=False)
val_rmse = mean_squared_error(y_val_, model.predict(x_val).reshape(-1), squared=False)

print(f"✅ (Optuna) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Optuna) Val   RMSE: {val_rmse:.4f}")

# 10. 제출
final_pred = model.predict(X_test_selected).reshape(-1)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/hybrid_optuna_submission.csv', index=False)
model.save('./_save/hybrid_optuna_model.h5')
print("📁 최종 제출 파일 및 모델 저장 완료")
