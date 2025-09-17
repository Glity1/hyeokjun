# 1. 라이브러리
import pandas as pd
import numpy as np
import joblib
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# 2. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')
y = train['Inhibition'].values

# 3. RDKit 피처 추출 함수
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

# 4. ChemBERTa 임베딩 추출 함수
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

# 5. RDKit 피처 생성
train_features = train['Canonical_Smiles'].apply(featurize)
test_features = test['Canonical_Smiles'].apply(featurize)
valid_idx = train_features.notnull()
train = train[valid_idx]
train_features = train_features[valid_idx]
y = y[valid_idx]

X_rdkit = np.array(train_features.tolist())
X_test_rdkit = np.array(test_features[test_features.notnull()].tolist())

# 6. Drop feature 적용
to_drop = joblib.load('./_save/final_drop_cols.pkl')
top_idx = joblib.load('./_save/final_top_idx.pkl')

X_rdkit_top = X_rdkit[:, top_idx]
X_test_rdkit_top = X_test_rdkit[:, top_idx]
X_rdkit_drop = np.delete(X_rdkit_top, to_drop, axis=1)
X_test_rdkit_drop = np.delete(X_test_rdkit_top, to_drop, axis=1)

# 7. ChemBERTa 임베딩 생성
train_chemberta = get_chemberta_embedding(train['Canonical_Smiles'].tolist())
test_chemberta = get_chemberta_embedding(test['Canonical_Smiles'].tolist())

# 8. 병합 및 스케일링
X_merge = np.concatenate([X_rdkit_drop, train_chemberta], axis=1)
X_test_merge = np.concatenate([X_test_rdkit_drop, test_chemberta], axis=1)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_merge)
X_test_scaled = scaler.transform(X_test_merge)

joblib.dump(scaler, './_save/chemberta_drop_scaler.pkl')

# 9. 학습/검증 분할
x_train, x_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 10. DNN 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 11. 학습
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=300,
          batch_size=64,
          callbacks=[es],
          verbose=1)

# 12. 평가
train_rmse = mean_squared_error(y_train, model.predict(x_train).reshape(-1), squared=False)
val_rmse = mean_squared_error(y_val, model.predict(x_val).reshape(-1), squared=False)

print(f"✅ (ChemBERTa + Drop) Train RMSE: {train_rmse:.4f}")
print(f"✅ (ChemBERTa + Drop) Val   RMSE: {val_rmse:.4f}")

# 13. 제출 및 저장
final_pred = model.predict(X_test_scaled).reshape(-1)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/chemberta_drop_submission.csv', index=False)
model.save('./_save/chemberta_drop_model.h5')
print("📁 제출 파일 및 모델 저장 완료")
