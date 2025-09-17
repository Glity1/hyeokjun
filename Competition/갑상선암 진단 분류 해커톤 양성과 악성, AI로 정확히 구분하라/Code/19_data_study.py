import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 경로 통일
path = './_data/dacon/cancer/'
save_path = './saved_model/'
sub_path = './_data/dacon/cancer/sub/'

# 데이터 불러오기
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

X = train.drop('Cancer', axis=1)
y = train['Cancer']

# 인코딩
binary_cols = ['Gender','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']
le = LabelEncoder()
for col in binary_cols:
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])

multi_cols = ['Country', 'Race']
X = pd.get_dummies(X, columns=multi_cols)
test = pd.get_dummies(test, columns=multi_cols)
missing_cols = set(X.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0
test = test[X.columns]

# 스케일링
scale_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
scaler = MinMaxScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
test[scale_cols] = scaler.transform(test[scale_cols])

# ✅ 고급 파생변수 추가 (최적화된)
epsilon = 1e-8
X['T4_TSH_Ratio'] = X['T4_Result'] / (X['TSH_Result'] + epsilon)
X['T3_T4_Ratio']  = X['T3_Result'] / (X['T4_Result'] + epsilon)
X['Age_Nodule_Interaction'] = X['Age'] * X['Nodule_Size']
X['T3_TSH_Ratio'] = X['T3_Result'] / (X['TSH_Result'] + epsilon)

test['T4_TSH_Ratio'] = test['T4_Result'] / (test['TSH_Result'] + epsilon)
test['T3_T4_Ratio']  = test['T3_Result'] / (test['T4_Result'] + epsilon)
test['Age_Nodule_Interaction'] = test['Age'] * test['Nodule_Size']
test['T3_TSH_Ratio'] = test['T3_Result'] / (test['TSH_Result'] + epsilon)

# 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# 딥러닝 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=300, batch_size=256, validation_split=0.1, callbacks=[es],
                 class_weight=class_weight_dict, verbose=2)

# Threshold 튜닝
y_proba = model.predict(X_valid).ravel()
prec, recall, thresholds = precision_recall_curve(y_valid, y_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + epsilon)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"최적 threshold: {best_threshold:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 제출 예측
test_pred_proba = model.predict(test).ravel()
test_pred = (test_pred_proba >= best_threshold).astype(int)
sub['Cancer'] = test_pred
sub.to_csv(sub_path + 'submission_final.csv', index=False)

# 모델 저장
model.save(save_path + 'thyroid_dl_final_model.h5')
model.save_weights(save_path + 'thyroid_dl_final_weights.h5')
print("모델 저장 완료")

# ✅ 전체 시각화
plt.figure(figsize=(8,6))
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.grid(); plt.title('Loss Curve')
plt.show()

cm = confusion_matrix(y_valid, (y_proba >= best_threshold).astype(int))
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix'); plt.show()

plt.figure(figsize=(7,5))
plt.plot(recall, prec)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.show()

fpr, tpr, _ = roc_curve(y_valid, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); plt.show()

plt.figure(figsize=(8,5))
sns.histplot(y_proba[y_valid==0], color='skyblue', label='Negative', stat='density', kde=True)
sns.histplot(y_proba[y_valid==1], color='salmon', label='Positive', stat='density', kde=True)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
plt.legend(); plt.title('Predicted Probability Distribution'); plt.show()
