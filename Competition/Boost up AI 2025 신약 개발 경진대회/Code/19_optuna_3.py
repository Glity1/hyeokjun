# 1. 라이브러리
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# 2. 데이터 로딩
path = './_data/dacon/new/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

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

# 4. featurize 학습 데이터
train_features = train['Canonical_Smiles'].apply(featurize)
train = train[train_features.notnull()]
train_features = train_features[train_features.notnull()]
X = np.array(train_features.tolist())
y = train['Inhibition'].values

# 5. 스케일링 + Top 50 피처 선택
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

xgb_base = XGBRegressor(random_state=42)
xgb_base.fit(X_scaled, y)

selector = SelectFromModel(xgb_base, threshold=-np.inf, max_features=50, prefit=True)
X_selected = selector.transform(X_scaled)

# 6. Test featurize
test_features = test['Canonical_Smiles'].apply(featurize)
test_features = test_features[test_features.notnull()]
X_test = np.array(test_features.tolist())
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# 7. 기존 모델로 Test 예측 → Confident 샘플 추출 (더 강화된 기준)
test_pred = xgb_base.predict(X_test_scaled)
mean = np.mean(test_pred)
std = np.std(test_pred)

lower = mean - 0.5 * std
upper = mean + 0.5 * std

confident_idx = np.where((test_pred >= lower) & (test_pred <= upper))[0]
confident_X = X_test_selected[confident_idx]
confident_y = test_pred[confident_idx]

print(f"✅ Confident Test 샘플 수 (강화 기준): {len(confident_y)}")

# 8. 기존 학습 데이터와 합치기
X_aug = np.concatenate([X_selected, confident_X], axis=0)
y_aug = np.concatenate([y, confident_y], axis=0)

# 9. 학습/검증 분할
x_train, x_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

# 10. 과적합 줄인 XGBoost 구성
xgb_final = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)

xgb_final.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)

# 11. 성능 평가
train_pred = xgb_final.predict(x_train)
val_pred = xgb_final.predict(x_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"✅ (XGB+Pseudo/Reg) Train RMSE: {train_rmse:.4f}")
print(f"✅ (XGB+Pseudo/Reg) Val   RMSE: {val_rmse:.4f}")

# 12. 최종 예측 및 저장
final_pred = xgb_final.predict(X_test_selected)
submit['Inhibition'] = final_pred
submit.to_csv('./_save/xgb_pseudo_regularized_submission.csv', index=False)
print("📁 제출 파일 저장 완료")

# 13. 저장
joblib.dump(xgb_final, './_save/xgb_pseudo_regularized_model.pkl')
joblib.dump(scaler, './_save/xgb_pseudo_regularized_scaler.pkl')
joblib.dump(selector, './_save/xgb_pseudo_regularized_selector.pkl')
