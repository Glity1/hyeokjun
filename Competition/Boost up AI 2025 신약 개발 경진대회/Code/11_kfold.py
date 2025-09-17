import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import os

# 1. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
test = pd.read_csv('./_data/dacon/new/test.csv')
submit = pd.read_csv('./_data/dacon/new/sample_submission.csv')

# 2. Feature 추출 함수 (2D + 3D)
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 3D 구조 생성
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    desc = {
        # 2D
        'MolLogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),

        # 3D
        'Asphericity': rdmd.CalcAsphericity(mol),
        'Eccentricity': rdmd.CalcEccentricity(mol),
        'InertialShapeFactor': rdmd.CalcInertialShapeFactor(mol),
        'NPR1': rdmd.CalcNPR1(mol),
        'NPR2': rdmd.CalcNPR2(mol),
        'RadiusOfGyration': rdmd.CalcRadiusOfGyration(mol),
        'SpherocityIndex': rdmd.CalcSpherocityIndex(mol)
    }

    return desc

# 3. Feature 추출
train_feats = train['Canonical_Smiles'].apply(featurize)
test_feats = test['Canonical_Smiles'].apply(featurize)

# 4. Drop None (결측 분자 제거)
train = train[train_feats.notnull()]
train_feats = train_feats[train_feats.notnull()]
train_feats = pd.DataFrame(train_feats.tolist()).reset_index(drop=True)
test_feats = pd.DataFrame(test_feats.tolist()).reset_index(drop=True)
y = train['Inhibition'].values

# 5. 스케일링
scaler = MinMaxScaler()
x = scaler.fit_transform(train_feats)
x_test = scaler.transform(test_feats)

# 6. K-Fold + EarlyStopping
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
    print(f"\n🔁 Fold {fold+1}")
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )

    val_pred = model.predict(x_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    val_scores.append(rmse)
    print(f"📉 Fold {fold+1} RMSE: {rmse:.4f}")

    test_preds += model.predict(x_test) / kf.n_splits

# 7. 전체 성능 출력
print("\n✅ 평균 Validation RMSE:", np.mean(val_scores))

# 8. 모델 저장
os.makedirs('./_save/xgb/', exist_ok=True)
joblib.dump(model, './_save/xgb/xgb_kfold_model.pkl')

# 9. 제출 저장
submit['Inhibition'] = test_preds
submit.to_csv('./_save/xgb/submission_kfold.csv', index=False)
print("📝 제출 파일 저장 완료")
