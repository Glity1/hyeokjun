import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 경로 설정
train_path = './_data/dacon/new/train.csv'
test_path = './_data/dacon/new/test.csv'
submit_path = './_data/dacon/new/sample_submission.csv'
save_dir = './_save/xgb_combined_fp_desc/'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 로드
train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)
submit = pd.read_csv(submit_path)
y = train_raw['Inhibition']

# ✅ RDKit Descriptor 전체 리스트
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# ✅ Descriptor + Fingerprint 추출 함수
def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return list(calc.CalcDescriptors(mol))

def get_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

def extract_features(smiles_list):
    desc_list = []
    fp_list = []
    for smi in smiles_list:
        desc = get_rdkit_descriptors(smi)
        fp = get_morgan_fp(smi)
        desc_list.append(desc)
        fp_list.append(fp)
    desc_df = pd.DataFrame(desc_list, columns=descriptor_names)
    fp_df = pd.DataFrame(fp_list, columns=[f'FP_{i}' for i in range(2048)])
    return pd.concat([desc_df, fp_df], axis=1)

# ✅ 피처 추출
X_train = extract_features(train_raw['Canonical_Smiles'])
X_test = extract_features(test_raw['Canonical_Smiles'])

# ✅ 결측치 처리
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ✅ 스케일링
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 모델 정의 및 학습
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=222,
    verbosity=0
)
model.fit(X_train_scaled, y)

# ✅ 모델 저장
model.save_model(os.path.join(save_dir, 'xgb_combined.model'))
joblib.dump(model, os.path.join(save_dir, 'xgb_combined.pkl'))

# ✅ 예측 및 평가
y_pred = model.predict(X_train_scaled)
rmse = mean_squared_error(y, y_pred, squared=False)
corr, _ = pearsonr(y, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"📈 Pearson Correlation (B): {corr:.4f}")

# ✅ 상관계수 시각화
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.5)
sns.regplot(x=y, y=y_pred, scatter=False, color='red')
plt.xlabel("True Inhibition")
plt.ylabel("Predicted Inhibition")
plt.title(f"Prediction vs True (B={corr:.4f})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_plot.png'))
plt.show()

# ✅ 제출 파일 생성
submit['Inhibition'] = model.predict(X_test_scaled)
submit.to_csv(os.path.join(save_dir, 'xgb_submission_combined.csv'), index=False)
print("📄 제출 파일 저장 완료.")
