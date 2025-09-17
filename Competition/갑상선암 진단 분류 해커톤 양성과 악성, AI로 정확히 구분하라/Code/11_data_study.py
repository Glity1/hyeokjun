import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로딩
path = './_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 라벨인코딩 (범주형 전부 적용)
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 3. 스케일링
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 4. train-test split + SMOTE
x_train, x_valid, y_train, y_valid = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

# 5. 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_sm), y=y_train_sm)
class_weight_dict = dict(enumerate(class_weights))

# 6. 모델 설계 (Adaptive Dropout)
model = Sequential()
model.add(Dense(128, input_dim=x_train_sm.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
hist = model.fit(x_train_sm, y_train_sm, epochs=300, batch_size=256, validation_split=0.2,
                 callbacks=[es], class_weight=class_weight_dict, verbose=2)

# 7. Validation에서 threshold 튜닝 (F1 최적)
y_proba = model.predict(x_valid).ravel()
precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
print(f"Best threshold: {best_thresh:.4f}, Best F1: {f1_scores[best_idx]:.4f}")

# 8. 최종 예측
final_pred = (y_proba >= best_thresh).astype(int)
cm = confusion_matrix(y_valid, final_pred)
print(cm)

# 9. 제출파일 생성
final_test_pred_proba = model.predict(test_scaled).ravel()
final_test_pred = (final_test_pred_proba >= best_thresh).astype(int)
submission_csv['Cancer'] = final_test_pred
submission_csv.to_csv(path + 'submission_adaptive_dropout.csv')
print("✅ 제출파일 저장 완료!")

# 10. 간단 시각화
plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.title('Loss Curve')
plt.legend()
plt.show()


from sklearn.metrics import f1_score

# 최종 validation F1 score 명확히 출력
final_f1 = f1_score(y_valid, final_pred)
print(f"Validation F1 score (확인용) : {final_f1:.4f}")