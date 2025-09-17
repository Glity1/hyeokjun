# ------------------------------------
# 0. 기본 라이브러리 준비
# ------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# focal loss 정의
def focal_loss(gamma=1.5):
    def loss(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean((1 - pt) ** gamma * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0)))
    return loss

# ------------------------------------
# 1. 데이터 준비 (같이 진행한 버전 기준)
# ------------------------------------
path = './_data/dacon/cancer/' 
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 인코딩 및 스케일링
binary_cols = ['Gender','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']
multi_cols = ['Country', 'Race']
le = LabelEncoder()

for col in binary_cols + multi_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

X = train.drop('Cancer', axis=1)
y = train['Cancer']

# 파생 변수 추가
X['T4_TSH_Ratio'] = X['T4_Result'] / (X['TSH_Result'] + 1e-6)
X['T3_T4_Ratio'] = X['T3_Result'] / (X['T4_Result'] + 1e-6)
X['Age_Nodule_Interaction'] = X['Age'] * X['Nodule_Size']

test['T4_TSH_Ratio'] = test['T4_Result'] / (test['TSH_Result'] + 1e-6)
test['T3_T4_Ratio'] = test['T3_Result'] / (test['T4_Result'] + 1e-6)
test['Age_Nodule_Interaction'] = test['Age'] * test['Nodule_Size']

scale_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
              'T4_TSH_Ratio', 'T3_T4_Ratio', 'Age_Nodule_Interaction']
scaler = MinMaxScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
test[scale_cols] = scaler.transform(test[scale_cols])

# 훈련/검증 분리
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# ------------------------------------
# 2. 딥러닝 모델 학습
# ------------------------------------
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss=focal_loss(gamma=1.5), optimizer='adam')
es = EarlyStopping(patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=256, validation_split=0.1, callbacks=[es], class_weight=class_weight_dict, verbose=0)

dl_pred_valid = model.predict(X_valid).ravel()
dl_pred_test = model.predict(test).ravel()

# ------------------------------------
# 3. 부스팅 모델 학습 (LightGBM)
# ------------------------------------
lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred_valid = lgb_model.predict_proba(X_valid)[:,1]
lgb_pred_test = lgb_model.predict_proba(test)[:,1]

# ------------------------------------
# 4. 앙상블 & Threshold 튜닝
# ------------------------------------
ensemble_valid = (dl_pred_valid + lgb_pred_valid) / 2

prec, recall, thresholds = precision_recall_curve(y_valid, ensemble_valid)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
print(f"최적 threshold: {best_thresh:.4f}, Validation F1 Score: {f1_scores[best_idx]:.4f}")

# ------------------------------------
# 5. 제출 파일 생성
# ------------------------------------
ensemble_test = (dl_pred_test + lgb_pred_test) / 2
final_pred = (ensemble_test >= best_thresh).astype(int)
sub['Cancer'] = final_pred
sub.to_csv(path + 'final_submission_ensemble.csv')
print("✅ 제출파일 생성 완료!")
