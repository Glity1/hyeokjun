# 1. 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 2. 데이터 불러오기
train = pd.read_csv('./_data/dacon/new/train.csv')
y = train['Inhibition'].values
smiles_list = train['Canonical_Smiles'].tolist()

# 3. ChemBERTa 임베딩 추출
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model_chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)

def get_chemberta_embedding(smiles_list):
    embeddings = []
    for smi in tqdm(smiles_list, desc="ChemBERTa"):
        with torch.no_grad():
            inputs = tokenizer(smi, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model_chemberta(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
    return np.array(embeddings)

X_chemberta = get_chemberta_embedding(smiles_list)

# 4. RDKit + Morgan 특성 추출
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
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

rdkit_feats = [featurize(smi) for smi in tqdm(smiles_list, desc="RDKit+Morgan")]
valid_idx = [i for i, feat in enumerate(rdkit_feats) if feat is not None]

# 5. 유효한 데이터로 정리
X_rdkit = np.array([rdkit_feats[i] for i in valid_idx])
X_chemberta = X_chemberta[valid_idx]
y_valid = y[valid_idx]

# 6. 특성 결합 + 정규화
X_all = np.concatenate([X_rdkit, X_chemberta], axis=1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"✅ 전체 특성 수: {X_scaled.shape[1]}")

# 7. 분산 기준 설명력 낮은 특성 제거
var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_scaled)
print(f"✅ 분산 기준 필터링 후: {X_var.shape}")

# 8. Feature Importance 시각화 (GradientBoosting 기준)
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_var, y_valid)

importances = gbr.feature_importances_
top_indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.bar(range(len(top_indices)), importances[top_indices])
plt.xticks(range(len(top_indices)), [f"F{i}" for i in top_indices], rotation=45)
plt.title("Top 20 Feature Importances (GradientBoosting)")
plt.tight_layout()
plt.show()

# 9. 상관관계 시각화 (Top 20 중요 특성)
top_X = X_var[:, top_indices]
top_corr = pd.DataFrame(top_X).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix (Top 20 Important Features)")
plt.tight_layout()
plt.show()

# 10. PCA 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_var)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_valid, cmap='viridis', s=10)
plt.colorbar(sc, label='Inhibition')
plt.title("PCA Projection (2D) of Hybrid Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
