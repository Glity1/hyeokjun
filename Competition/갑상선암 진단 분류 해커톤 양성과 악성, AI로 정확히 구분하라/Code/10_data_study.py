# ================== 라이브러리 ==================
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

# SMOTE를 위한 imbalanced-learn 라이브러리 추가
from imblearn.over_sampling import SMOTE

# 시각화를 위한 라이브러리 추가
import matplotlib.pyplot as plt
import seaborn as sns

# 부스팅 모델
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier 

# 그래프 설정 (한글 깨짐 방지, 음수 부호 깨짐 방지 등)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준
plt.rcParams['axes.unicode_minus'] = False # 음수 부호 깨짐 방지
sns.set_style('whitegrid') # Seaborn 스타일 설정

# ================== 1. 데이터 로딩 및 전처리 ==================
path = './_data/dacon/cancer/'

model_save_path = path + 'saved_models/' # 모델 저장 경로 정의
sub_save_path = './_submit/cancer/' # 제출 파일 저장 경로 추가

# 모델 저장 폴더가 없으면 생성
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(sub_save_path, exist_ok=True) # 제출 파일 폴더 생성
print(f"모델 저장 경로: {model_save_path}")
print(f"제출 파일 저장 경로: {sub_save_path}")

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print("Train 데이터 결측치:\n", train_csv.isnull().sum()[train_csv.isnull().sum() > 0])
print("\nTest 데이터 결측치:\n", test_csv.isnull().sum()[test_csv.isnull().sum() > 0])

# exit() # 이 부분 제거

# --- 원본 데이터의 연속형 컬럼 통계량 확인 (초반) ---
print("##### 연속형 컬럼 원본 데이터 통계량 (전처리 전) #####")
continuous_cols_initial = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
print(train_csv[continuous_cols_initial].describe())
print("\n")

# 타겟 변수 분리
x = train_csv.drop('Cancer', axis=1).copy()
y = train_csv['Cancer'].copy()
test_csv_processed = test_csv.copy()


# Label Encoding (2개의 고유값을 가지는 이진 컬럼)
le = LabelEncoder()
for col in ['Gender','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    x[col] = le.fit_transform(x[col])
    test_csv_processed[col] = le.transform(test_csv_processed[col])
print("이진 범주형 컬럼 Label Encoding 완료.")


# One-Hot Encoding (순서가 없는 다중 범주형 컬럼)
ohe_cols = ['Country', 'Race']
combined_df = pd.concat([x, test_csv_processed], axis=0) # train, test 합쳐서 OHE
combined_df = pd.get_dummies(combined_df, columns=ohe_cols, drop_first=False, dtype=int) # dtype=int 추가

# 다시 train과 test 데이터셋으로 분리
x = combined_df.iloc[:len(train_csv)].copy()
test_csv_processed = combined_df.iloc[len(train_csv):].copy()

# OHE 후 컬럼 순서 일치 (중요)
train_cols = x.columns
test_cols = test_csv_processed.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    test_csv_processed[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    x[c] = 0
    
x = x[train_cols] # 순서 통일
test_csv_processed = test_csv_processed[train_cols] # 순서 통일
print(f"다중 범주형 컬럼 One-Hot Encoding 완료. x.shape: {x.shape}, test_csv_processed.shape: {test_csv_processed.shape}")


# --- 2단계: 연속형 변수 전처리 (수정 반영) ---
print("\n##### 2단계: 연속형 변수 전처리 (MinMaxScaler 적용) #####")

# 2-1. 분포 변환 (로그 변환) 대상 컬럼 정의 및 적용 -> 균등 분포로 판단되어 제거
# 2-2. 이상치(Outlier) 처리 (IQR 방식의 Winsorization) -> 뚜렷한 이상치 없어 제거
# 2-3. 'Age' 컬럼 구간화 (Binning) -> 균등 분포이므로 제거, 원래 Age 컬럼 사용

# 모든 연속형 변수 스케일링 (MinMaxScaler 사용)
continuous_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
scaler = MinMaxScaler()

x[continuous_cols] = scaler.fit_transform(x[continuous_cols])
test_csv_processed[continuous_cols] = scaler.transform(test_csv_processed[continuous_cols])
print(f"연속형 컬럼 {continuous_cols}에 MinMaxScaler 적용 완료.")

# 2-4. 연속형 변수들 간의 다중공선성 (상관관계) 확인
print("\n##### 연속형 변수들 간의 상관관계 (다중공선성) 확인 (스케일링 후) #####")
# 스케일링 후에도 상관계수 값은 변하지 않으므로, 이 부분은 그대로 유지
corr_matrix = x[continuous_cols].corr() 
print("\n상관관계 매트릭스:\n", corr_matrix)

print("\n--- 2단계 전처리 완료 (로그 변환/이상치 처리/Age 구간화 제거) ---")


# ================== 3단계: 특성 공학 (Feature Engineering) 시작 ==================
print("\n##### 3단계: 특성 공학 (Feature Engineering) 시작 #####")

# 작은 상수 (epsilon) 정의: 0으로 나누는 오류 방지
epsilon = 1e-6

# 아이디어 1: 갑상선 호르몬 비율 특성 (Ratio Features)
# 스케일링된 값으로 비율 특성 생성
x['T4_TSH_Ratio'] = x['T4_Result'] / (x['TSH_Result'] + epsilon)
test_csv_processed['T4_TSH_Ratio'] = test_csv_processed['T4_Result'] / (test_csv_processed['TSH_Result'] + epsilon)
print("'T4_TSH_Ratio' 생성 완료.")

x['T3_T4_Ratio'] = x['T3_Result'] / (x['T4_Result'] + epsilon)
test_csv_processed['T3_T4_Ratio'] = test_csv_processed['T3_Result'] / (test_csv_processed['T4_Result'] + epsilon)
print("'T3_T4_Ratio' 생성 완료.")

# 아이디어 2: 특정 기준치 초과/미달 여부 (Binary Features)
# 스케일링된 값에 대한 임계치 고려 (예: 스케일링 후 0.5가 중간값 정도 될 수 있음)
# 여기서는 원본 데이터의 일반적인 정상 범위 기준을 스케일링 후 값으로 변환하여 사용하거나,
# 스케일링된 값의 평균/중앙값을 기준으로 설정하는 것이 합리적입니다.
# 일단은 스케일링된 값의 평균으로 기준을 잡겠습니다.
tsh_mean_scaled = x['TSH_Result'].mean()
t4_mean_scaled = x['T4_Result'].mean()
t3_mean_scaled = x['T3_Result'].mean()

x['TSH_High'] = (x['TSH_Result'] > tsh_mean_scaled).astype(int)
test_csv_processed['TSH_High'] = (test_csv_processed['TSH_Result'] > tsh_mean_scaled).astype(int)
print("'TSH_High' 생성 완료.")

x['T4_Low'] = (x['T4_Result'] < t4_mean_scaled).astype(int)
test_csv_processed['T4_Low'] = (test_csv_processed['T4_Result'] < t4_mean_scaled).astype(int)
print("'T4_Low' 생성 완료.")

x['T3_High'] = (x['T3_Result'] > t3_mean_scaled).astype(int)
test_csv_processed['T3_High'] = (test_csv_processed['T3_Result'] > t3_mean_scaled).astype(int)
print("'T3_High' 생성 완료.")

# 아이디어 3: 연령-결절 크기 상호작용 (Interaction Feature)
# 스케일링된 Age와 Nodule_Size 사용
x['Age_Nodule_Interaction'] = x['Age'] * x['Nodule_Size']
test_csv_processed['Age_Nodule_Interaction'] = test_csv_processed['Age'] * test_csv_processed['Nodule_Size']
print("'Age_Nodule_Interaction' 생성 완료.")

# Family_Background와 Nodule_Size 상호작용 (Family_Background는 LabelEncoded 되어있음)
x['Family_Background_Nodule_Interaction'] = x['Family_Background'] * x['Nodule_Size']
test_csv_processed['Family_Background_Nodule_Interaction'] = test_csv_processed['Family_Background'] * test_csv_processed['Nodule_Size']
print("'Family_Background_Nodule_Interaction' 생성 완료.")

print("\n--- 3단계: 특성 공학 (Feature Engineering) 완료 ---")
print(f"최종 특성 개수 (x): {x.shape[1]}개")
print(f"최종 특성 개수 (test_csv_processed): {test_csv_processed.shape[1]}개")
print("이제 모델 학습을 진행합니다.")

# ================== SMOTE 적용 및 클래스 가중치 재조정 ==================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=714, shuffle=True, stratify=y
)

# SMOTE 적용 전에 클래스 분포 확인
print(f"\n--- SMOTE 적용 전 y_train 클래스 분포:\n{pd.Series(y_train).value_counts()}")

# SMOTE 적용
smote = SMOTE(random_state=714)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# SMOTE 적용 후 클래스 분포 확인
print(f"\n--- SMOTE 적용 후 y_train_smote 클래스 분포:\n{pd.Series(y_train_smote).value_counts()}")

# 스케일링 (전체 데이터에 대한 fit_transform을 다시 적용하지 않도록 주의)
# 이미 위에서 MinMaxSclaer를 `x`와 `test_csv_processed`에 `fit_transform` 및 `transform` 해두었으므로,
# train_test_split으로 나뉜 `x_train_smote`, `x_test`, `test_csv_processed`를 다시 스케일링 해야 합니다.
# 중요한 것은, `scaler` 객체는 위에서 `x`를 이용해 `fit`된 상태여야 합니다.
# 따라서, 여기서는 `transform`만 진행합니다.
# NOTE: 만약 SMOTE 후 재스케일링이 필요하다고 판단되면,
# `scaler.fit_transform(x_train_smote)` 후 `scaler.transform(x_test)`를 해야 하지만,
# SMOTE는 기존 데이터 분포에 맞춰 새로운 샘플을 생성하므로,
# 원본 데이터에 fit된 스케일러로 변환하는 것이 일반적입니다.
# 현재 코드는 SMOTE 적용 후 `x_train_smote`에 `fit_transform`하고, `x_test`와 `test_csv_processed`에는 `transform`하는 방식으로 되어 있습니다.
# 이는 `MinMaxScaler`가 0과 1 사이로 압축하기 때문에 SMOTE로 생성된 데이터가 0-1 범위를 벗어날 가능성이 적어 괜찮을 수 있습니다.
# 만약 `StandardScaler`였다면 `fit_transform`을 다시 고려해봐야 합니다.

# 현재 코드의 스케일링 로직을 그대로 따르되, 주석을 통해 목적 명확화
scaler_for_smote = MinMaxScaler() # 새로운 스케일러 인스턴스 생성
x_train_scaled = scaler_for_smote.fit_transform(x_train_smote) # SMOTE가 적용된 데이터에 fit_transform
x_test_scaled = scaler_for_smote.transform(x_test) # test 데이터는 transform만
test_csv_scaled = scaler_for_smote.transform(test_csv_processed) # 최종 예측을 위한 test 데이터도 transform만


# SMOTE 적용 후 클래스 가중치 재계산 (기본 balanced)
class_weights_smote_base = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
class_weight_dict_smote = dict(enumerate(class_weights_smote_base))

# F1-Score 최적화를 위해 weight_multiplier 초기값을 1.0으로 설정
# 1.0은 'balanced'와 동일한 효과 (SMOTE 후 0.5/0.5). 필요시 1.5, 2.0 등으로 늘려가며 실험
weight_multiplier = 1.3 # 이전 1.28 -> 다시 1.0으로 초기화
class_weight_dict_smote[1] *= weight_multiplier 

print(f"\nClass weights (SMOTE 적용 후, 양성 클래스 가중치 조정): {class_weight_dict_smote}")


# ================== 4. 딥러닝 모델 ==================
model = Sequential()
model.add(Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint_filepath_dl = model_save_path + 'best_dl_model.h5'
mc_dl = ModelCheckpoint(
    filepath=checkpoint_filepath_dl,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1)

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True) 
model.fit(x_train_scaled, y_train_smote, # SMOTE가 적용된 데이터 사용
    epochs=500, # 에포크 증가. patience가 늘었으니 에포크도 증가
    batch_size=32, validation_split=0.2,
    callbacks=[es, mc_dl], class_weight=class_weight_dict_smote, verbose=2) # 조정된 클래스 가중치 사용

# 딥러닝 예측 확률 및 threshold 조정 (x_test_scaled 사용)
y_proba_dl = model.predict(x_test_scaled).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_dl)

# 딥러닝 모델 임계값을 F1-Score 최대화로 변경
f1_scores_dl = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx_dl = np.argmax(f1_scores_dl)
best_thresh_dl = thresholds[best_idx_dl]

y_pred_dl = (y_proba_dl >= best_thresh_dl).astype(int)

print(f"DL Best Threshold (F1-Score Maximize): {best_thresh_dl:.4f}")
print(f"DL Accuracy: {accuracy_score(y_test, y_pred_dl):.4f}")
print(f"DL F1 Score: {f1_score(y_test, y_pred_dl):.4f}")

best_dl_recall_at_thresh = recall[best_idx_dl]
best_dl_precision_at_thresh = precision[best_idx_dl]

print(f"DL Recall: {best_dl_recall_at_thresh:.4f}")
print(f"DL Precision: {best_dl_precision_at_thresh:.4f}")


# ================== 5. 부스팅 모델 ==================
# 양성 클래스 가중치 계산 (SMOTE 적용 후의 분포 기반)
num_neg_smote = y_train_smote.value_counts()[0]
num_pos_smote = y_train_smote.value_counts()[1]
# 부스팅 모델 가중치도 weight_multiplier 1.0으로 초기화
boosting_weight_for_pos = (num_neg_smote / num_pos_smote) * weight_multiplier # 딥러닝과 동일한 배율 사용

# XGBoost 모델 (SMOTE가 적용된 데이터 사용)
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss', # 경고 방지 및 일반적인 로스 지표 설정
    scale_pos_weight=boosting_weight_for_pos # 조정된 가중치 적용
)
xgb.fit(x_train_scaled, y_train_smote)
y_pred_xgb = xgb.predict(x_test_scaled) # 기본 임계값 0.5 사용 예측
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGB F1 Score:", f1_score(y_test, y_pred_xgb))
print("XGB Recall:", f1_score(y_test, y_pred_xgb, average=None)[1])

# LightGBM 모델 (SMOTE가 적용된 데이터 사용)
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=boosting_weight_for_pos, # LightGBM도 scale_pos_weight를 사용
    random_state=42
)
lgb_model.fit(x_train_scaled, y_train_smote)
y_pred_lgb = lgb_model.predict(x_test_scaled) # 기본 임계값 0.5 사용 예측
print("LGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LGBM F1 Score:", f1_score(y_test, y_pred_lgb))
print("LGBM Recall:", f1_score(y_test, y_pred_lgb, average=None)[1])

# CatBoost 모델 (SMOTE가 적용된 데이터 사용)
cat_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6, verbose=0,
    random_state=42,
    # CatBoost의 class_weights를 조정된 weight_multiplier 기반으로 변경
    class_weights=[1.0, boosting_weight_for_pos] # CatBoost는 클래스 가중치 리스트로 받음
)
cat_model.fit(x_train_scaled, y_train_smote)
y_pred_cat = cat_model.predict(x_test_scaled) # 기본 임계값 0.5 사용 예측
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost F1 Score:", f1_score(y_test, y_pred_cat))
print("CatBoost Recall:", f1_score(y_test, y_pred_cat, average=None)[1])


# ================== 6. 앙상블 (수동 Soft Voting) ==================
print("\n##### 앙상블 (수동 Soft Voting) 시작 - 모든 모델 포함 #####")

# 각 모델의 test_csv_scaled에 대한 예측 확률 얻기 (최종 제출용)
test_proba_dl = model.predict(test_csv_scaled).ravel()
test_proba_xgb = xgb.predict_proba(test_csv_scaled)[:, 1]
test_proba_lgb = lgb_model.predict_proba(test_csv_scaled)[:, 1]
test_proba_cat = cat_model.predict_proba(test_csv_scaled)[:, 1]

# test_csv_scaled에 대한 모든 모델의 예측 확률 평균
test_proba_ensemble = (test_proba_dl + test_proba_xgb + test_proba_lgb + test_proba_cat) / 4

# 각 모델의 x_test_scaled에 대한 예측 확률 얻기 (threshold 조정을 위함)
y_proba_dl_test = model.predict(x_test_scaled).ravel()
y_proba_xgb_test = xgb.predict_proba(x_test_scaled)[:, 1]
y_proba_lgb_test = lgb_model.predict_proba(x_test_scaled)[:, 1]
y_proba_cat_test = cat_model.predict_proba(x_test_scaled)[:, 1]

# 모든 모델의 예측 확률 평균 (Soft Voting)
y_proba_ensemble = (y_proba_dl_test + y_proba_xgb_test + y_proba_lgb_test + y_proba_cat_test) / 4 

# 앙상블 임계값을 F1-Score 최대화로 변경
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_ensemble)

f1_scores_ensemble = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx_ensemble = np.argmax(f1_scores_ensemble)
best_thresh_ensemble = thresholds[best_idx_ensemble]

y_pred_ensemble = (y_proba_ensemble >= best_thresh_ensemble).astype(int)

print(f"Ensemble Best Threshold (F1-Score Maximize): {best_thresh_ensemble:.4f}")
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
print(f"Ensemble F1 Score: {f1_score(y_test, y_pred_ensemble):.4f}")

best_ensemble_recall_at_thresh = recall[best_idx_ensemble]
best_ensemble_precision_at_thresh = precision[best_idx_ensemble]

print(f"Ensemble Recall: {best_ensemble_recall_at_thresh:.4f}")
print(f"Ensemble Precision: {best_ensemble_precision_at_thresh:.4f}")


# ================== 8. 모델 성능 시각화 ==================
print("\n##### 8. 앙상블 모델 성능 시각화 #####")

# 8-1. Confusion Matrix (혼동 행렬)
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('앙상블 모델 Confusion Matrix')
plt.show()
print(f"Confusion Matrix:\n{cm}")
print(f"True Negative (TN): {cm[0,0]}")
print(f"False Positive (FP): {cm[0,1]}")
print(f"False Negative (FN): {cm[1,0]}")
print(f"True Positive (TP): {cm[1,1]}")


# 8-2. ROC Curve 및 AUC
fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('앙상블 모델 ROC Curve')
plt.legend(loc="lower right")
plt.show()
print(f"앙상블 모델 ROC AUC: {roc_auc:.4f}")

# 8-3. Precision-Recall Curve
pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba_ensemble)
plt.figure(figsize=(7, 6))
plt.plot(pr_recall, pr_precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall (재현율)')
plt.ylabel('Precision (정밀도)')
plt.title('앙상블 모델 Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
plt.show()

# 8-4. 클래스별 예측 확률 분포 (히스토그램)
plt.figure(figsize=(10, 6))
sns.histplot(y_proba_ensemble[y_test == 0], color='skyblue', label='Actual Negative (0)', kde=True, stat="density", linewidth=0)
sns.histplot(y_proba_ensemble[y_test == 1], color='salmon', label='Actual Positive (1)', kde=True, stat="density", linewidth=0)
plt.axvline(best_thresh_ensemble, color='red', linestyle='--', label=f'Optimal Threshold ({best_thresh_ensemble:.2f})')
plt.title('앙상블 모델 예측 확률 분포 (클래스별)')
plt.xlabel('예측 확률 (양성 클래스)')
plt.ylabel('밀도')
plt.legend()
plt.show()

# ================== 9. 모델 및 가중치 저장 ==================
print("\n##### 9. 모델 및 가중치 저장 #####")

# 9-1. 딥러닝 모델 저장
import joblib # joblib으로 XGBoost도 저장 가능
print(f"딥러닝 모델 (최적 가중치) 저장 완료: {checkpoint_filepath_dl}")

# 9-2. XGBoost 모델 저장
xgb_save_path = model_save_path + 'xgb_model.json' # JSON 형식으로 저장
xgb.save_model(xgb_save_path)
print(f"XGBoost 모델 저장 완료: {xgb_save_path}")

# 9-3. LightGBM 모델 저장
lgbm_save_path_joblib = model_save_path + 'lgbm_model.pkl'
joblib.dump(lgb_model, lgbm_save_path_joblib)
print(f"LightGBM 모델 저장 완료: {lgbm_save_path_joblib}")

# 9-4. CatBoost 모델 저장
cat_save_path = model_save_path + 'catboost_model.cbm'
cat_model.save_model(cat_save_path)
print(f"CatBoost 모델 저장 완료: {cat_save_path}")

print("모든 모델 및 가중치 저장 완료!")

# ================== 7. 제출 파일 생성 (수동 Soft Voting 기반) ==================
final_test_pred = (test_proba_ensemble >= best_thresh_ensemble).astype(int)

submission_csv['Cancer'] = final_test_pred
# 제출 파일명에 F1-Score 최적화 전략 반영
submission_csv.to_csv(sub_save_path + 'submission_f1_score_optimized_v4.csv') # 저장 경로 변경 및 버전 업데이트
print(f"✅ {sub_save_path}submission_f1_score_optimized_v4.csv 파일 생성 완료!")