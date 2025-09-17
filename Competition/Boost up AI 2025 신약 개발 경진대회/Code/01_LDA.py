import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ✅ 저장 경로
save_dir = './_save/dnn_lda_models/'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 로딩
path = './_data/dacon/new/'
train_raw = pd.read_csv(path + 'train.csv')
test_raw = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ RDKit descriptor 추출
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumRings': Descriptors.RingCount(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'MolWt_TPSA': Descriptors.MolWt(mol) * Descriptors.TPSA(mol),
        'Donor/Acceptor': Descriptors.NumHDonors(mol) / (Descriptors.NumHAcceptors(mol) + 1),
        'RotBond2': Descriptors.NumRotatableBonds(mol) ** 2
    }

# ✅ 피처 생성
train_features = train_raw['Canonical_Smiles'].apply(featurize)
test_features = test_raw['Canonical_Smiles'].apply(featurize)
train_df = pd.DataFrame(train_features.tolist())
test_df = pd.DataFrame(test_features.tolist())
train_df['Inhibition'] = train_raw['Inhibition']

# ✅ 컬럼 제거
drop_cols = ['ExactMolWt', 'MolWt_TPSA', 'RotBond2']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# ✅ log1p + 이진화
log_cols = ['MolWt', 'TPSA', 'HeavyAtomCount']
for col in log_cols:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])

bin_cols = ['NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
for col in bin_cols:
    train_df[f'{col}_bin'] = (train_df[col] > 0).astype(int)
    test_df[f'{col}_bin'] = (test_df[col] > 0).astype(int)

# ✅ 입력/타겟 분리
X = train_df.drop(columns=['Inhibition'])
y = train_df['Inhibition']
X_test = test_df.copy()

# ✅ RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ y를 LDA용 클래스 라벨로 변환 (5개 클래스)
y_bins = pd.qcut(y, q=5, labels=False)

# ✅ LDA 적용 (n_components = 클래스 수 - 1 = 4)
lda = LinearDiscriminantAnalysis(n_components=4)
X_lda = lda.fit_transform(X_scaled, y_bins)
X_test_lda = lda.transform(X_test_scaled)

# ✅ DNN 모델 정의
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='huber', optimizer='adam')
    return model

# ✅ KFold 학습
n_splits = 3
seed = 222
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
test_preds = np.zeros(len(X_test_lda))
valid_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_lda)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_val = X_lda[train_idx], X_lda[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = build_model(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=300, batch_size=32,
              callbacks=[es], verbose=1)

    # 모델 저장
    model_path = os.path.join(save_dir, f'dnn_lda_fold{fold+1}.h5')
    model.save(model_path)
    print(f"💾 저장 완료: {model_path}")

    val_pred = model.predict(X_val).flatten()
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    valid_scores.append(rmse)
    print(f"✅ Fold {fold+1} RMSE: {rmse:.5f}")

    test_preds += model.predict(X_test_lda).flatten() / n_splits

# ✅ 제출 저장
print(f"\n🎯 평균 RMSE (LDA 4D): {np.mean(valid_scores):.5f}")
submit['Inhibition'] = test_preds
submit.to_csv(path + 'dnn_submission_lda.csv', index=False)
print("📄 저장 완료: dnn_submission_lda.csv")


# 평균 RMSE (LDA 4D): 24.92663
