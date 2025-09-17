import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import lightgbm as lgb
import joblib
import os

# 🔧 1. 데이터 로드
path = './_data/dacon/cancer/'      
train = pd.read_csv(path + 'train.csv', index_col=0)   
test = pd.read_csv(path + 'test.csv', index_col=0) 
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 🔧 2. 라벨인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])

# 🔧 3. 파생변수 생성
def create_features(df):
    eps = 1e-8
    df['T4_TSH_Ratio'] = df['T4_Result'] / (df['TSH_Result'] + eps)
    df['T3_T4_Ratio']  = df['T3_Result'] / (df['T4_Result'] + eps)
    df['T3_TSH_Ratio'] = df['T3_Result'] / (df['TSH_Result'] + eps)
    df['Age_Nodule_Interaction'] = df['Age'] * df['Nodule_Size']
    return df

train = create_features(train)
test = create_features(test)

X = train.drop(['Cancer'], axis=1)
y = train['Cancer']

# 🔧 4. 데이터 분할 및 스케일링
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)

# 🔧 5. Class Weight 적용
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# 🔧 6. 딥러닝 모델 정의
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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

# 모델저장
model_path = './saved_models/'
os.makedirs(model_path, exist_ok=True)
checkpoint = ModelCheckpoint(model_path + 'model_v3.h5', save_best_only=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=300, batch_size=256,
    callbacks=[es, checkpoint],
    class_weight=class_weight_dict
)

# 🔧 7. LightGBM 훈련
lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
lgb_val = lgb.Dataset(X_val_scaled, label=y_val)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'seed': 42,
}

lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=300, early_stopping_rounds=30, verbose_eval=50)

# LGBM 저장
lgb_model.save_model(model_path + 'lgb_model_v3.txt')

# 🔧 8. 앙상블 예측 (validation 기준)
dl_val_pred = model.predict(X_val_scaled).ravel()
lgb_val_pred = lgb_model.predict(X_val_scaled)

ensemble_val_pred = (dl_val_pred + lgb_val_pred) / 2

# 🔧 9. 최적 threshold 탐색
prec, rec, thresholds = precision_recall_curve(y_val, ensemble_val_pred)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"\nBest threshold: {best_thresh:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 🔧 10. 평가
val_pred_final = (ensemble_val_pred >= best_thresh).astype(int)
cm = confusion_matrix(y_val, val_pred_final)
print(cm)

# 🔧 11. 최종 test셋 예측 및 제출파일 생성
dl_test_pred = model.predict(test_scaled).ravel()
lgb_test_pred = lgb_model.predict(test_scaled)
test_ensemble_pred = (dl_test_pred + lgb_test_pred) / 2
final_test_pred = (test_ensemble_pred >= best_thresh).astype(int)

sub['Cancer'] = final_test_pred
sub.to_csv('./submissions/submission_v3.csv', index=False)

print("✅ 제출파일 저장 완료")

# 🔧 12. 시각화
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss Curve"); plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix"); plt.show()

plt.figure(figsize=(7,5))
plt.plot(rec, prec, label="Precision-Recall Curve")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()

fpr, tpr, _ = roc_curve(y_val, ensemble_val_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.legend(); plt.title("ROC Curve"); plt.show()
