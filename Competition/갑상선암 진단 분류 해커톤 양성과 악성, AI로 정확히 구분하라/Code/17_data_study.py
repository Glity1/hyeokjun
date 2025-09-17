import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# 1. 데이터 로드
path = './_data/dacon/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 파생변수 생성
for df in [train, test]:
    df['T4_TSH_Ratio'] = df['T4_Result'] / (df['TSH_Result']+1e-6)
    df['T3_T4_Ratio'] = df['T3_Result'] / (df['T4_Result']+1e-6)
    df['Age_Nodule_Interaction'] = df['Age'] * df['Nodule_Size']

# 3. 전처리
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

x = train.drop(['Cancer'], axis=1)
y = train['Cancer']

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
test_scaled = scaler.transform(test)

# 4. 클래스 가중치 계산 (딥러닝용)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 5. 모델 설계
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=256,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=0
)

# 6. Threshold 튜닝
val_pred_prob = model.predict(x_val).ravel()
prec, recall, thresholds = precision_recall_curve(y_val, val_pred_prob)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"최적 threshold: {best_threshold:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 7. 제출 파일 생성
final_test_pred = (model.predict(test_scaled).ravel() >= best_threshold).astype(int)
submission['Cancer'] = final_test_pred
submission.to_csv(path + 'submission_final_total.csv')

# 8. 시각화
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

val_pred = (val_pred_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(7,5))
plt.plot(recall, prec)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.figure(figsize=(7,5))
plt.plot(thresholds, f1_scores[:-1])
plt.axvline(best_threshold, color='r', linestyle='--')
plt.title('Threshold vs F1 Curve')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.show()
