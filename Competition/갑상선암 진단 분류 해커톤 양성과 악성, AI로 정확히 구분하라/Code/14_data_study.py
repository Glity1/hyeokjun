import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
path = './_data/dacon/cancer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 파생변수 생성
epsilon = 1e-6
for df in [train_csv, test_csv]:
    df['T4_TSH_Ratio'] = df['T4_Result'] / (df['TSH_Result'] + epsilon)
    df['T3_T4_Ratio'] = df['T3_Result'] / (df['T4_Result'] + epsilon)
    df['Age_Nodule_Interaction'] = df['Age'] * df['Nodule_Size']

# 3. 인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

# 4. feature / target 분리
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 5. train/test 분할 (stratify 유지)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=777, shuffle=True, stratify=y
)

# 6. 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 7. 클래스 가중치 (positive 가중치 강화)
class_weight_dict = {0: 1, 1: 2.0}   # FN 보완

# 8. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 9. 컴파일 및 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

hist = model.fit(
    x_train, y_train, 
    epochs=300, batch_size=256,
    validation_split=0.1,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

# 10. 평가 및 threshold 튜닝
y_proba = model.predict(x_test).ravel()
prec, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold for F1: {best_threshold:.4f}")
print(f"Best F1 score on validation: {f1_scores[best_idx]:.4f}")

y_predict = (y_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 11. 제출 파일 생성
test_pred_proba = model.predict(test_csv_scaled).ravel()
test_pred = (test_pred_proba >= best_threshold).astype(int)

submission_csv['Cancer'] = test_pred
submission_csv.to_csv(path + 'submission_final_stable.csv')

# 12. 시각화
plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss Curve")
plt.show()

plt.plot(thresholds, f1_scores[:-1])
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
plt.title("Threshold vs F1")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.show()
