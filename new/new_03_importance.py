import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 저장 경로
save_dir = './_save/dnn_pca_models/'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 불러오기
path = './_data/dacon/new/'
train_raw = pd.read_csv(path + 'train.csv')
test_raw = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ RDKit 피처 추출
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

# ✅ 불필요 컬럼 제거
drop_cols = ['ExactMolWt', 'MolWt_TPSA', 'RotBond2']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# ✅ log1p 변환
log_cols = ['MolWt', 'TPSA', 'HeavyAtomCount']
for col in log_cols:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])

# ✅ 이진화
bin_cols = ['NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
for col in bin_cols:
    train_df[f'{col}_bin'] = (train_df[col] > 0).astype(int)
    test_df[f'{col}_bin'] = (test_df[col] > 0).astype(int)

# ✅ 스케일링
X = train_df.drop(columns=['Inhibition'])
y = train_df['Inhibition']
X_test = test_df.copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ PCA 적용
n_pca_components = 13
pca = PCA(n_components=n_pca_components)
X_scaled_pca = pca.fit_transform(X_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

# ✅ 모델 정의
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(loss='huber', optimizer='adam')
    return model

# ✅ KFold 학습
n_splits = 3
seed = 222
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
test_preds = np.zeros(len(X_test_scaled_pca))
valid_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled_pca)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_val = X_scaled_pca[train_idx], X_scaled_pca[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = build_model(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=300, batch_size=32,
              callbacks=[es], verbose=1)

    # 모델 저장
    model_path = os.path.join(save_dir, f'dnn_pca_fold{fold+1}.h5')
    model.save(model_path)
    print(f"💾 모델 저장 완료: {model_path}")

    val_pred = model.predict(X_val).flatten()
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    valid_scores.append(rmse)
    print(f"✅ Fold {fold+1} RMSE: {rmse:.5f}")

    test_preds += model.predict(X_test_scaled_pca).flatten() / n_splits

    # ✅ Permutation Importance (fold 1만 계산)
    if fold == 0:
        from sklearn.linear_model import Ridge
        ridge = Ridge().fit(X_train, y_train)  # surrogate model
        result = permutation_importance(ridge, X_val, y_val, scoring='neg_root_mean_squared_error', n_repeats=10, random_state=seed)
        importance_df = pd.DataFrame({
            'Feature': [f'PC{i+1}' for i in range(X_train.shape[1])],
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)
        
        # 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title(f'Permutation Importance (Fold {fold+1}) on PCA Features')
        plt.tight_layout()
        plt.savefig(path + f'permutation_importance_pca_fold{fold+1}.png')
        plt.show()

# ✅ 제출 저장
print(f"\n🎯 평균 RMSE (PCA {n_pca_components}D): {np.mean(valid_scores):.5f}")
submit['Inhibition'] = test_preds
submit.to_csv(path + f'dnn_submission_pca{n_pca_components}.csv', index=False)
print(f"📄 저장 완료: dnn_submission_pca{n_pca_components}.csv")
