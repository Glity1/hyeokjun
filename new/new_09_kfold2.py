import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors as rdmd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
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
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return {
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
        'SpherocityIndex': rdmd.CalcSpherocityIndex(mol),
    }

# 3. 특성 추출
train_feats = train['Canonical_Smiles'].apply(featurize)
test_feats = test['Canonical_Smiles'].apply(featurize)

# 4. Drop None
train = train[train_feats.notnull()]
train_feats = train_feats[train_feats.notnull()]
train_feats = pd.DataFrame(train_feats.tolist()).reset_index(drop=True)
test_feats = pd.DataFrame(test_feats.tolist()).reset_index(drop=True)
y = train['Inhibition'].values

# 5. Scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(train_feats)
x_test = scaler.transform(test_feats)

# 6. SelectFromModel
xgb_base = XGBRegressor(n_estimators=100, random_state=42)
xgb_base.fit(x, y)
selector = SelectFromModel(xgb_base, prefit=True, max_features=15, threshold=-np.inf)
x_selected = selector.transform(x)
x_test_selected = selector.transform(x_test)
selected_features = train_feats.columns[selector.get_support()]
print("✅ 선택된 특성:", selected_features.tolist())

# 7. KFold + 모델 학습 + 시각화용 데이터 저장
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_scores, train_scores = [], []
test_preds = np.zeros(len(test))
all_val_pred, all_val_true = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_selected)):
    print(f"\n🔁 Fold {fold+1}")
    x_train, x_val = x_selected[train_idx], x_selected[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(x_train, y_train,
              eval_set=[(x_val, y_val)],
              early_stopping_rounds=50,
              verbose=0)

    val_pred = model.predict(x_val)
    train_pred = model.predict(x_train)

    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    train_rmse = mean_squared_error(y_train, train_pred, squared=False)

    val_scores.append(val_rmse)
    train_scores.append(train_rmse)
    test_preds += model.predict(x_test_selected) / kf.n_splits

    all_val_pred.extend(val_pred)
    all_val_true.extend(y_val)

    print(f"📉 Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

# 8. 결과 저장
os.makedirs('./_save/xgb/', exist_ok=True)
submit['Inhibition'] = test_preds
submit.to_csv('./_save/xgb/submission_final.csv', index=False)
joblib.dump(model, './_save/xgb/xgb_final_model.pkl')
print("\n✅ 평균 Val RMSE:", np.mean(val_scores))

# 9. 과적합 시각화: Fold별 RMSE
plt.figure(figsize=(8, 5))
plt.plot(train_scores, label='Train RMSE', marker='o')
plt.plot(val_scores, label='Val RMSE', marker='o')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Train vs Validation RMSE by Fold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/xgb/rmse_fold_plot.png')
plt.show()

# 10. 예측 vs 실제 산점도
plt.figure(figsize=(6, 6))
plt.scatter(all_val_true, all_val_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Validation Prediction vs Actual')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/xgb/pred_vs_actual.png')
plt.show()

# 11. SHAP 해석
explainer = shap.Explainer(model, x_selected, feature_names=selected_features)
shap_values = explainer(x_selected)
shap.summary_plot(shap_values, features=x_selected, feature_names=selected_features)
