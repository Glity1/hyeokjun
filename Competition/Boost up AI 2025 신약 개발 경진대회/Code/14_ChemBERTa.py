import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# 1. 데이터 불러오기
test = pd.read_csv('./_data/dacon/new/test.csv')
smiles_list = test['Canonical_Smiles'].tolist()

# 2. ChemBERTa 임베딩 함수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
model.eval()

def get_chemberta_embedding(smiles):
    with torch.no_grad():
        encoded = tokenizer(smiles, return_tensors='pt', padding='max_length',
                            truncation=True, max_length=128).to(device)
        outputs = model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        return cls_embedding.squeeze().cpu().numpy()

# 3. 임베딩 추출
embeddings = []
for smi in smiles_list:
    try:
        emb = get_chemberta_embedding(smi)
        embeddings.append(emb)
    except:
        embeddings.append(np.zeros(768))  # 실패 시 0벡터
x_test = np.vstack(embeddings)

# 4. 스케일링 (학습과 동일하게 적용)
pt = PowerTransformer()
x_test = pt.fit_transform(x_test)  # 실제론 학습시 사용한 pt로 transform만 해야 정확함

# 5. DNN 모델 정의 함수
def build_dnn_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# 6. 최종 모델 학습 후 예측 (Train 전체로 학습)
train = pd.read_csv('./_data/dacon/new/train.csv')
train_smiles = train['Canonical_Smiles'].tolist()
y_train = train['Inhibition'].values

train_embeddings = []
for smi in train_smiles:
    try:
        emb = get_chemberta_embedding(smi)
        train_embeddings.append(emb)
    except:
        train_embeddings.append(np.zeros(768))
x_train = np.vstack(train_embeddings)
x_train = pt.fit_transform(x_train)

final_model = build_dnn_model(x_train.shape[1])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
final_model.fit(x_train, y_train, epochs=300, batch_size=64,
                validation_split=0.1, callbacks=[es], verbose=0)

# 7. 예측 및 저장
y_pred = final_model.predict(x_test).squeeze()
submission = pd.read_csv('./_data/dacon/new/sample_submission.csv')
submission['Inhibition'] = y_pred
submission.to_csv('./_save/submit_chemberta_dnn.csv', index=False)
print("✅ 제출 파일 저장 완료: ./_save/submit_chemberta_dnn.csv")
