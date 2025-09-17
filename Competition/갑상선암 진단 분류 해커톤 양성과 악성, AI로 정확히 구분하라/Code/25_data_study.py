import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 데이터 로드
path = './_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 라벨 인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

# 3. 파생 변수 추가
eps = 1e-8
for df in [train_csv, test_csv]:
    df['Age_Nodule'] = df['Age'] * df['Nodule_Size']
    df['T3_T4_Ratio']  = df['T3_Result'] / (df['T4_Result'] + eps)
    df['T4_TSH_Ratio'] = df['T4_Result'] / (df['TSH_Result'] + eps)
    df['T3_TSH_Ratio'] = df['T3_Result'] / (df['TSH_Result'] + eps)

# 4. 피처/타겟 분리
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 5. 데이터 분할 (seed 222)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=222, stratify=y)

# 6. 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 7. 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 8. 모델 구성
model = Sequential([
    Dense(64, input_dim=x_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 9. 모델 저장 경로
save_path = './_save/keras29_final_seed222.h5'

# 10. 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=400, batch_size=128,
                 validation_split=0.2, callbacks=[es],
                 class_weight=class_weight_dict, verbose=2)

model.save(save_path)

# 11. 평가 및 threshold 찾기
y_proba = model.predict(x_test).ravel()
prec, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + eps)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f}")

# 12. 최종 예측 및 점수
y_pred = (y_proba >= best_threshold).astype(int)
print("Validation F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 13. 제출 파일 생성
test_proba = model.predict(test_csv_scaled).ravel()
test_pred = (test_proba >= best_threshold).astype(int)
submission_csv['Cancer'] = test_pred
submission_csv.to_csv('./_save/submission_charge_seed222.csv')
