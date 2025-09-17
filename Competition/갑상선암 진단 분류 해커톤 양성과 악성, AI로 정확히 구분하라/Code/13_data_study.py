import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
path = './_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. LabelEncoding (딥러닝에 적합)
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History',
            'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

# 3. 피처 엔지니어링
epsilon = 1e-6
train_csv['T4_TSH_Ratio'] = train_csv['T4_Result'] / (train_csv['TSH_Result'] + epsilon)
test_csv['T4_TSH_Ratio'] = test_csv['T4_Result'] / (test_csv['TSH_Result'] + epsilon)

train_csv['T3_T4_Ratio'] = train_csv['T3_Result'] / (train_csv['T4_Result'] + epsilon)
test_csv['T3_T4_Ratio'] = test_csv['T3_Result'] / (test_csv['T4_Result'] + epsilon)

train_csv['Age_Nodule_Interaction'] = train_csv['Age'] * train_csv['Nodule_Size']
test_csv['Age_Nodule_Interaction'] = test_csv['Age'] * test_csv['Nodule_Size']

# 4. X, y 분리
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 5. train/val 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

# 6. 스케일링 (MinMax)
scale_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
              'T4_TSH_Ratio','T3_T4_Ratio','Age_Nodule_Interaction']
scaler = MinMaxScaler()
x_train[scale_cols] = scaler.fit_transform(x_train[scale_cols])
x_test[scale_cols] = scaler.transform(x_test[scale_cols])
test_csv[scale_cols] = scaler.transform(test_csv[scale_cols])

# 7. 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 8. 딥러닝 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)

history = model.fit(
    x_train, y_train, 
    epochs=500, batch_size=128,
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

# 9. Threshold 튜닝 (F1 기준 최적)
y_proba = model.predict(x_test).ravel()
prec, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"최적 threshold: {best_threshold:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 10. 테스트셋 예측 및 제출 파일 생성
test_proba = model.predict(test_csv).ravel()
test_pred = (test_proba >= best_threshold).astype(int)

submission_csv['Cancer'] = test_pred
submission_csv.to_csv(path + 'final_dl_submission.csv', index=True)
print("✅ 제출파일 생성 완료!")

# 11. 성능 시각화
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(pd.crosstab(y_test, (y_proba >= best_threshold).astype(int)), 
            annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(7,5))
plt.plot(recall, prec, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# 예측 확률 (여기에 본인 모델 결과 넣으세요)
y_pred_proba = model.predict(x_test).ravel()

# Precision-Recall 계산
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

# 최적 threshold 구하기
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# 시각화
plt.figure(figsize=(8,6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold ({best_thresh:.2f})')
plt.title('Precision & Recall vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

print(f"최적 Threshold: {best_thresh:.4f}, 최적 F1: {best_f1:.4f}")