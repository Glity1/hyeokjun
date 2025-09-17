import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
path = './_data/dacon/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 라벨 인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History',
            'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 3. 파생 변수 생성
train['T4_TSH'] = train['T4_Result'] / (train['TSH_Result']+1e-5)
test['T4_TSH'] = test['T4_Result'] / (test['TSH_Result']+1e-5)

train['T3_T4'] = train['T3_Result'] / (train['T4_Result']+1e-5)
test['T3_T4'] = test['T3_Result'] / (test['T4_Result']+1e-5)

train['Age_Nodule'] = train['Age'] * train['Nodule_Size']
test['Age_Nodule'] = test['Age'] * test['Nodule_Size']

X = train.drop(['Cancer'], axis=1)
y = train['Cancer']

# 스케일링
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# 4. Focal Loss 정의 (gamma=1.5)
def focal_loss(gamma=1.5):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean((1 - pt) ** gamma * K.log(pt))
    return loss

# 5. 초기 모델 학습
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss=focal_loss(gamma=1.5), optimizer=Adam(0.001))
es = EarlyStopping(patience=20, restore_best_weights=True)

hist = model.fit(X_train, y_train, epochs=200, batch_size=256,
                 validation_data=(X_val, y_val), callbacks=[es],
                 class_weight=class_weight_dict, verbose=0)

# 6. Pseudo Label 적용
pseudo_pred = model.predict(test_scaled).ravel()
pseudo_threshold = 0.75  # 0.7~0.8 중 0.75 선택

pseudo_idx = np.where((pseudo_pred > pseudo_threshold) | (pseudo_pred < 1-pseudo_threshold))[0]
pseudo_X = test_scaled[pseudo_idx]
pseudo_y = (pseudo_pred[pseudo_idx] >= 0.5).astype(int)

# 기존 학습 데이터 + Pseudo 합치기
X_pseudo = np.vstack([X, pseudo_X])
y_pseudo = np.hstack([y, pseudo_y])

# 7. 최종 모델 재학습
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_pseudo, y_pseudo, test_size=0.2, stratify=y_pseudo, random_state=42)

model_final = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_final.compile(loss=focal_loss(gamma=1.5), optimizer=Adam(0.001))
es = EarlyStopping(patience=20, restore_best_weights=True)

hist_final = model_final.fit(X_train_final, y_train_final, epochs=200, batch_size=256,
                              validation_data=(X_val_final, y_val_final),
                              callbacks=[es], class_weight=class_weight_dict, verbose=0)

# 8. Threshold 튜닝
val_pred_proba = model_final.predict(X_val_final).ravel()
prec, recall, thresholds = precision_recall_curve(y_val_final, val_pred_proba)
f1_scores = 2*(prec*recall)/(prec+recall+1e-8)
best_idx = np.argmax(f1_scores)
best_th = thresholds[best_idx]

print(f"최적 threshold: {best_th:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 9. 제출
test_pred = model_final.predict(test_scaled).ravel()
final_submit = (test_pred >= best_th).astype(int)
submission['Cancer'] = final_submit
submission.to_csv(path + 'submission_final_ultra.csv')

# 10. 시각화 유지
plt.plot(hist_final.history['loss'], label='Train Loss')
plt.plot(hist_final.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()
plt.show()

conf = confusion_matrix(y_val_final, (val_pred_proba >= best_th).astype(int))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

plt.plot(recall, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

plt.plot(thresholds, f1_scores[:-1])
plt.axvline(best_th, color='r', linestyle='--')
plt.title("Threshold vs F1 Curve")
plt.show()
