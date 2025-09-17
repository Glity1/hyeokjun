import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# ✅ 기존 RDKit Top20 특성
top20_features = [
    'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors',
    'HeavyAtomCount', 'FractionCSP3', 'RingCount', 'NHOHCount', 'NOCount',
    'NumAliphaticRings', 'NumAromaticRings', 'NumSaturatedRings', 'ExactMolWt',
    'MolMR', 'LabuteASA', 'BalabanJ', 'MaxPartialCharge', 'MinPartialCharge'
]

# ✅ 추가 후보 특성 불러오기
df_add = pd.read_csv('./_data/dacon/new/rdkit_additional_top_features.csv')  # Feature 컬럼에 있음
additional_features = df_add['Feature'].tolist()

# ✅ 전체 특성 = 기존 + 추가
all_features = top20_features + additional_features

# ✅ SMILES 리스트 (학습 데이터)
train = pd.read_csv('./_data/dacon/new/train.csv').drop_duplicates('Canonical_Smiles').reset_index(drop=True)
smiles_list = train['Canonical_Smiles'].tolist()

# ✅ RDKit Descriptor Function Mapping
desc_funcs = {name: getattr(Descriptors, name) for name in all_features if hasattr(Descriptors, name)}

# ✅ 특성 계산
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        values = []
        for name in desc_funcs:
            val = desc_funcs[name](mol)
            values.append(np.nan_to_num(val))
        return values
    except:
        return None

# ✅ 전체 DataFrame 생성
X = []
valid_smiles = []
for sm in smiles_list:
    row = compute_descriptors(sm)
    if row:
        X.append(row)
        valid_smiles.append(sm)

X = pd.DataFrame(X, columns=list(desc_funcs.keys()))
X = X.fillna(0)  # 혹시 모를 NaN 제거

# ✅ 상관 행렬 계산
corr_matrix = X.corr().abs()

# ✅ 상관계수 기준 중복 제거
selected = list(top20_features)  # 기존은 무조건 포함
dropped = []

for feat in additional_features:
    if feat not in X.columns:
        continue
    high_corr = False
    for base_feat in top20_features:
        if base_feat in X.columns:
            corr = corr_matrix.loc[feat, base_feat]
            if corr >= 0.9:
                high_corr = True
                break
    if not high_corr:
        selected.append(feat)
    else:
        dropped.append(feat)

# ✅ 결과 출력
print(f"🎯 최종 선택된 RDKit 특성 수: {len(selected)}개")
print(f"❌ 제거된 중복 특성 수: {len(dropped)}개")

# ✅ 선택된 Descriptor 이름들 저장
with open('./_save/selected_rdkit_features.txt', 'w') as f:
    for name in selected:
        f.write(name + '\n')
