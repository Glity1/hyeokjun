import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer, AutoModel
import torch

# 1. 데이터 로드
train = pd.read_csv('./_data/dacon/new/train.csv')
smiles_list = train['Canonical_Smiles'].tolist()

# 2. ChemBERTa 임베딩
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

chemberta_features = get_chemberta_embedding(smiles_list)

# 3. RDKit + Morgan
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
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

features = [featurize(smi) for smi in smiles_list]
valid_idx = [i for i, f in enumerate(features) if f is not None]
features = [features[i] for i in valid_idx]
chemberta_features = chemberta_features[valid_idx]

# 4. 통합
df_fp_desc = pd.DataFrame(features)
df_chemberta = pd.DataFrame(chemberta_features)
df_all = pd.concat([df_fp_desc, df_chemberta], axis=1)
df_all.columns = [f'f_{i}' for i in range(df_all.shape[1])]

# 5. 상관관계 히트맵
corr = df_all.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("SMILES 기반 전체 특성 상관관계 히트맵")
plt.tight_layout()
plt.show()

# 6. 높은 상관관계 특성쌍 추출
high_corr_pairs = []
threshold = 0.95
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[1]):
        if abs(corr.iloc[i, j]) > threshold:
            high_corr_pairs.append((df_all.columns[i], df_all.columns[j], corr.iloc[i, j]))

# 결과 출력
print(f"\n📌 상관계수 |r| > {threshold} 이상인 특성 쌍:")
for f1, f2, r in high_corr_pairs:
    print(f"{f1} ↔ {f2} : corr = {r:.3f}")
