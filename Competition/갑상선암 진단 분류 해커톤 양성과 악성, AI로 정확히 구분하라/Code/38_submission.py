import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# 부스팅 모델 임포트
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. 데이터
path = './_data/dacon/cancer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# Label Encoding (Country, Race 포함)
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# train_test_split 시 stratify=y 적용 (클래스 분포 유지)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=777,
    shuffle=True,
    stratify=y
)

# 스케일링
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 클래스 가중치 계산 (훈련 데이터 기준)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights for Deep Learning model:", class_weight_dict)

# 부스팅 모델을 위한 양성 클래스 가중치 (음성 샘플 수 / 양성 샘플 수)
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight_value = num_neg / num_pos
print(f"scale_pos_weight for Boosting models: {scale_pos_weight_value:.2f}")

# 2. 딥러닝 모델 구성 및 훈련
print("\n--- Deep Learning Model Training ---")
dl_model = Sequential()
dl_model.add(Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'))
dl_model.add(BatchNormalization())
dl_model.add(Dropout(0.1))
dl_model.add(Dense(64, activation='relu'))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(64, activation='relu'))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dropout(0.3))
dl_model.add(Dense(1, activation='sigmoid'))

# 컴파일, 훈련
dl_model.compile(loss='binary_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])

es_dl = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True,
)
hist_dl = dl_model.fit(
    x_train_scaled, y_train,
    epochs=400, batch_size=64,
    validation_split=0.1,
    callbacks=[es_dl],
    class_weight=class_weight_dict,
    verbose=1 # 학습 과정을 간결하게 표시
)
print("Deep Learning model training complete.")

# 3. 부스팅 모델 구성 및 훈련

# XGBoost
print("\n--- XGBoost Model Training ---")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, # 경고 메시지 방지
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight_value, # 클래스 불균형 처리
    random_state=777,
    tree_method='hist' # 대용량 데이터셋에서 더 빠른 학습을 위해
)
# XGBoost는 EarlyStopping을 fit 인자로 직접 전달
xgb_model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)], # 검증 세트
              early_stopping_rounds=50, # 50라운드 동안 성능 개선 없으면 조기 종료
              verbose=False) # 학습 과정을 간결하게 표시
print("XGBoost model training complete.")

# LightGBM
print("\n--- LightGBM Model Training ---")
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1, # 제한 없음
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight_value, # 클래스 불균형 처리
    random_state=777
)
lgb_model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]) # LightGBM의 EarlyStopping
print("LightGBM model training complete.")

# CatBoost
print("\n--- CatBoost Model Training ---")
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='F1', # F1을 직접 모니터링
    class_weights=[1.0, scale_pos_weight_value], # 클래스 불균형 처리
    random_seed=777,
    verbose=False # 학습 과정을 간결하게 표시
)
cat_model.fit(x_train, y_train,
              eval_set=(x_test, y_test),
              early_stopping_rounds=50)
print("CatBoost model training complete.")

# 4. 개별 모델 예측 (확률)
print("\n--- Generating Predictions ---")
# 딥러닝 모델은 스케일링된 데이터 사용
dl_pred_proba = dl_model.predict(x_test_scaled).ravel()

# 부스팅 모델은 스케일링되지 않은 원본 데이터 사용 (MinMaxScaler는 트리 모델에 필수는 아님)
xgb_pred_proba = xgb_model.predict_proba(x_test)[:, 1]
lgb_pred_proba = lgb_model.predict_proba(x_test)[:, 1]
cat_pred_proba = cat_model.predict_proba(x_test)[:, 1]

# 5. 앙상블 (Soft Voting - 단순 평균)
# 필요에 따라 가중치를 부여할 수 있으나, 여기서는 단순 평균
ensemble_pred_proba = (dl_pred_proba + xgb_pred_proba + lgb_pred_proba + cat_pred_proba) / 4

# 앙상블 예측에 대한 최적 threshold 찾기
prec, recall, thresholds = precision_recall_curve(y_test, ensemble_pred_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold_ensemble = thresholds[best_idx]

print(f"\nEnsemble Best F1 Score: {f1_scores[best_idx]:.4f}")
print(f"Ensemble Best Threshold: {best_threshold_ensemble:.4f}")

ensemble_y_predict = (ensemble_pred_proba >= best_threshold_ensemble).astype(int)

ensemble_f1 = f1_score(y_test, ensemble_y_predict)
ensemble_acc = accuracy_score(y_test, ensemble_y_predict)
print(f"Ensemble F1 Score on Test Set: {ensemble_f1:.4f}")
print(f"Ensemble Accuracy Score on Test Set: {ensemble_acc:.4f}")

# 6. 테스트셋 예측 및 제출
print("\n--- Generating Submission File ---")
# 개별 모델 테스트셋 예측
dl_test_pred_proba = dl_model.predict(test_csv_scaled).ravel()
xgb_test_pred_proba = xgb_model.predict_proba(test_csv)[:, 1]
lgb_test_pred_proba = lgb_model.predict_proba(test_csv)[:, 1]
cat_test_pred_proba = cat_model.predict_proba(test_csv)[:, 1]

# 앙상블 테스트셋 예측
ensemble_test_pred_proba = (dl_test_pred_proba + xgb_test_pred_proba + lgb_test_pred_proba + cat_test_pred_proba) / 4
final_test_pred = (ensemble_test_pred_proba >= best_threshold_ensemble).astype(int)

submission_csv['Cancer'] = final_test_pred
submission_csv.to_csv(path + 'submission_ensemble_0612_final.csv')
print("Submission file created: submission_ensemble_0612_final.csv")

# 7. 시각화 (딥러닝 모델의 학습 곡선만)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(hist_dl.history['loss'], c='red', label='loss')
plt.plot(hist_dl.history['val_loss'], c='blue', label='val_loss')
plt.title('Deep Learning Model Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()