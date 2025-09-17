# -------------------- [기존 코드 포함] -------------------- #
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
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
path = './_data/dacon/cancer/'
save_path = './_save/keras23/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 라벨 인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History',
            'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 3. 데이터 분할 및 스케일링
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_scaled = scaler.transform(test_csv)

# 4. 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# 5. 모델 구성
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

# 6. 콜백 및 학습
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
history = model.fit(
    x_train, y_train,
    epochs=400, batch_size=128,
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

# 7. 평가 및 threshold 찾기
y_proba = model.predict(x_test).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 8. 예측 및 지표 출력
y_pred = (y_proba >= best_threshold).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 9. 테스트셋 예측 및 저장
test_proba = model.predict(test_scaled).ravel()
test_pred = (test_proba >= best_threshold).astype(int)
submission_csv['Cancer'] = test_pred
submission_file = os.path.join(save_path, 'submission_seed222.csv')
submission_csv.to_csv(submission_file)
print("제출 파일 저장 완료:", submission_file)

# 10. 모델 저장
model_file = os.path.join(save_path, 'model_seed222.h5')
model.save(model_file)
print("모델 저장 완료:", model_file)

# -------------------- [시각화 섹션 추가] -------------------- #

# 1. Loss Graph
plt.figure(figsize=(9,6))
plt.plot(history.history['loss'], c='red', label='loss')
plt.plot(history.history['val_loss'], c='blue', label='val_loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 2. Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 3. Precision-Recall Curve
plt.figure(figsize=(7,5))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. 예측 확률 분포
plt.figure(figsize=(8,5))
sns.histplot(y_proba[y_test==0], color='skyblue', label='Negative', stat='density', kde=True)
sns.histplot(y_proba[y_test==1], color='salmon', label='Positive', stat='density', kde=True)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.show()
