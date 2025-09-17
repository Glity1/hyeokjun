import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import os

# ◼️ 설정
os.makedirs('./_save', exist_ok=True)

# ◼️ 데이터 로딩
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
smiles_list = train['Canonical_Smiles'].tolist()
y = train['Inhibition'].values

# ◼️ 기존 사용된 RDKit Top20 특성
used_features = [
    'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors',
    'HeavyAtomCount', 'FractionCSP3', 'RingCount', 'NHOHCount', 'NOCount',
    'NumAliphaticRings', 'NumAromaticRings', 'NumSaturatedRings', 'ExactMolWt',
    'MolMR', 'LabuteASA', 'BalabanJ', 'MaxPartialCharge', 'MinPartialCharge'
]

# ◼️ 전체 RDKit Descriptor 목록 불러오기
all_features = [desc[0] for desc in Descriptors._descList]
remaining_features = [f for f in all_features if f not in used_features]

# ◼️ 남은 RDKit 특성 계산 함수
def calc_remaining_features(smiles_list):
    data = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            data.append([0]*len(remaining_features))
            continue
        row = []
        for name in remaining_features:
            try:
                val = getattr(Descriptors, name)(mol)
            except:
                val = 0
            row.append(val)
        data.append(row)
    return np.array(data)

# ◼️ 특성 계산 및 스케일링
X_remaining = calc_remaining_features(smiles_list)
X_scaled = StandardScaler().fit_transform(np.nan_to_num(X_remaining))

# ◼️ 중요도 기반 선택
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
selector = SelectFromModel(rf, threshold='median', prefit=True)
selected_indices = selector.get_support(indices=True)
selected_names = [remaining_features[i] for i in selected_indices]

# ◼️ 결과 저장 및 출력
selected_df = pd.DataFrame({
    'Index': selected_indices,
    'Feature': selected_names,
    'Importance': rf.feature_importances_[selected_indices]
}).sort_values(by='Importance', ascending=False)

selected_df.to_csv('./_save/rdkit_additional_top_features.csv', index=False)
print("\n✅ 선택된 중요 특성:")
print(selected_df)

# ◼️ 선택된 추가 RDKit 특성을 새 모델에 적용하고 싶을 경우 아래 결과를 사용하면 됩니다:
# - `selected_names`: 중요도 높은 TopN 특성 이름
# - `calc_remaining_features(smiles_test)` 결과 중 해당 컬럼만 사용

# ★ 후속으로 모델에 반영할 때는 기존 RDKit 20개 + 이 추가 TopN을 concat해서 global feature로 사용하면 됩니다
