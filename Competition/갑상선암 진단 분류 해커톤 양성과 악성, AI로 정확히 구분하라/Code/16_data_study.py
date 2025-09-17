import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. 데이터 로드
path = './_data/dacon/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 라벨 인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 3. 파생변수 생성 (최적화 파생변수 버전)
epsilon = 1e-6
train['T4_TSH_Ratio'] = train['T4_Result'] / (train['TSH_Result'] + epsilon)
test['T4_TSH_Ratio'] = test['T4_Result'] / (test['TSH_Result'] + epsilon)

train['T3_T4_Ratio'] = train['T3_Result'] / (train['T4_Result'] + epsilon)
test['T3_T4_Ratio'] = test['T3_Result'] / (test['T4_Result'] + epsilon)

train['Age_Nodule_Interaction'] = train['Age'] * train['Nodule_Size']
test['Age_Nodule_Interaction'] = test['Age'] * test['Nodule_Size']

# 4. 스케일링
x = train.drop('Cancer', axis=1)
y = train['Cancer']

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test)

# 5. 클래스 가중치
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# 6. 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 7. 모델링
model = Sequential()
model.add(Dense(64, input_shape=(x_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[es],
    verbose=0
)

# 8. Threshold 튜닝
y_val_pred_proba = model.predict(x_val).ravel()
prec, rec, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"최적 threshold: {best_thresh:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 9. 시각화 - 전체 그래프 유지판

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Confusion matrix
y_val_pred = (y_val_pred_proba >= best_thresh).astype(int)
cm = confusion_matrix(y_val, y_val_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Precision-Recall Curve
plt.figure()
plt.plot(rec, prec, label="Precision-Recall Curve")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

# Threshold vs F1
plt.figure()
plt.plot(thresholds, f1_scores[:-1], label="F1 vs Threshold")
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold ({best_thresh:.2f})')
plt.title("Threshold vs F1 Curve")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

# 10. 최종 제출
test_pred_proba = model.predict(test_scaled).ravel()
test_pred = (test_pred_proba >= best_thresh).astype(int)
sub['Cancer'] = test_pred
sub.to_csv(path + 'submission_final_full.csv')
print("✅ 제출파일 생성 완료!")
