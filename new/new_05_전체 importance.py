import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
seed = 222
n_splits = 3
path = './_data/dacon/new/'
save_dir = './_save/final_feature_selection/'
os.makedirs(save_dir, exist_ok=True)
np.random.seed(seed)

# ✅ 데이터 불러오기
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# ✅ 피처 추출 함수: Morgan FP + RDKit
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Morgan Fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)

    # RDKit Descriptor
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.MolWt(mol) * Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol) / (Descriptors.NumHAcceptors(mol) + 1),
        Descriptors.NumRotatableBonds(mol) ** 2
    ]
    return np.concatenate([fp_array, desc])

# ✅ 피처 생성
train_features = train['Canonical_Smiles'].apply(featurize)
test_features = test['Canonical_Smiles'].apply(featurize)
X = np.vstack(train_features)
X_test = np.vstack(test_features)
y = train['Inhibition']

# ✅ 스케일링
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ 1단계: Variance Threshold
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X_scaled)
X_test_vt = vt.transform(X_test_scaled)

# ✅ 2단계: Feature Importance 기반 선택
rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
rf.fit(X_vt, y)
sfm = SelectFromModel(rf, threshold="mean", prefit=True)
X_sel = sfm.transform(X_vt)
X_test_sel = sfm.transform(X_test_vt)

# ✅ 최종 데이터 형태
print(f"원본 피처 수: {X.shape[1]}, Variance 제거 후: {X_vt.shape[1]}, 중요도 선택 후: {X_sel.shape[1]}")

# ✅ 모델 학습 (XGBoost)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
test_preds = np.zeros(len(X_test_sel))
valid_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_sel)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_val = X_sel[train_idx], X_sel[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, verbosity=0)
    model.fit(X_train, y_train)

    # 저장
    model_path = os.path.join(save_dir, f'xgb_fold{fold+1}.pkl')
    joblib.dump(model, model_path)
    print(f"💾 저장 완료: {model_path}")

    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"✅ Fold {fold+1} RMSE: {rmse:.5f}")
    valid_scores.append(rmse)

    test_preds += model.predict(X_test_sel) / n_splits

# ✅ 제출
submit['Inhibition'] = test_preds
submit_path = os.path.join(save_dir, 'submission_xgb_selected.csv')
submit.to_csv(submit_path, index=False)
print(f"\n🎯 평균 RMSE: {np.mean(valid_scores):.5f}")
print(f"📄 제출 파일 저장 완료: {submit_path}")

# ✅ Feature Importance 시각화
importances = rf.feature_importances_
selected_indices = sfm.get_support(indices=True)
selected_importances = importances[selected_indices]
feature_idx_sorted = np.argsort(selected_importances)[::-1]

plt.figure(figsize=(10, 8))
sns.barplot(x=selected_importances[feature_idx_sorted][:30],
            y=[f'F{i}' for i in selected_indices[feature_idx_sorted[:30]]])
plt.title('📊 Top 50 Important Features (RF based)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_top30.png'))
plt.show()
