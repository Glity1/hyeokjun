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

# 모델 저장 폴더가 없으면 생성
os.makedirs(model_save_path, exist_ok=True)
print(f"모델 저장 경로: {model_save_path}")

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print("Train 데이터 결측치:\n", train_csv.isnull().sum()[train_csv.isnull().sum() > 0])
print("\nTest 데이터 결측치:\n", test_csv.isnull().sum()[test_csv.isnull().sum() > 0])

exit()

# --- 원본 데이터의 연속형 컬럼 통계량 확인 (초반) ---
print("##### 연속형 컬럼 원본 데이터 통계량 (전처리 전) #####")
continuous_cols_initial = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
print(train_csv[continuous_cols_initial].describe())
print("\n")

# Label Encoding (2개의 고유값을 가지는 이진 컬럼)
le = LabelEncoder()
for col in ['Gender','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

# One-Hot Encoding (순서가 없는 다중 범주형 컬럼)
combined_df = pd.concat([train_csv.drop('Cancer', axis=1), test_csv], axis=0)
ohe_cols = ['Country', 'Race']
combined_df = pd.get_dummies(combined_df, columns=ohe_cols, drop_first=False)

# 다시 train과 test 데이터셋으로 분리
x = combined_df.iloc[:len(train_csv)].copy()
test_csv_processed = combined_df.iloc[len(train_csv):].copy()

# 'Cancer' 타겟 변수는 그대로 사용
y = train_csv['Cancer']

# --- 2단계: 연속형 변수 전처리 적용 시작 ---

# 2-1. 분포 변환 (로그 변환) 대상 컬럼 정의 (★★★ 변경 사항: 로그 변환 컬럼 활성화 및 추가 ★★★)
# Nodule_Size와 TSH_Result는 분포가 왜곡되었을 가능성이 높아 로그 변환 적용 시도
log_transform_cols = ['Nodule_Size', 'TSH_Result'] 

# --- 로그 변환 전 시각화 및 통계량 ---
# print("\n##### 로그 변환 전 각 컬럼의 분포 및 이상치 시각화 #####") # 필요시 주석 해제하여 확인
# for col in continuous_cols_initial:
#     print(f"\n--- 컬럼: '{col}' (로그 변환 전) ---")
#     plt.figure(figsize=(14, 5))
#     plt.subplot(1, 2, 1)
#     sns.histplot(x[col], kde=True)
#     plt.title(f'{col} (Before Log Transform)')
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=x[col])
#     plt.title(f'{col} Boxplot (Before Log Transform)')
#     plt.tight_layout()
#     plt.show()
#     print(f"{col} (Before Log Transform) describe:\n", x[col].describe())

# 실제 로그 변환 적용
for col in log_transform_cols:
    # 0이거나 음수인 값에 np.log1p를 적용하기 전에 작은 값 더하기 또는 조건부 처리
    # Nodule_Size, TSH_Result는 음수가 아닐 것이므로, 0에 대한 처리만 고려
    x[col] = np.log1p(x[col]) # log(1+x)
    test_csv_processed[col] = np.log1p(test_csv_processed[col])
    print(f"'{col}' 컬럼에 로그 변환 적용 완료.")

# --- 로그 변환 후 시각화 및 통계량 ---
# print("\n##### 로그 변환 후 (적용된 컬럼) 분포 시각화 #####") # 필요시 주석 해제하여 확인
# if log_transform_cols: 
#     for col in log_transform_cols:
#         print(f"\n--- 컬럼: '{col}' (로그 변환 후) ---")
#         plt.figure(figsize=(14, 5))
#         plt.subplot(1, 2, 1)
#         sns.histplot(x[col], kde=True)
#         plt.title(f'{col} (After Log Transform)')
#         plt.subplot(1, 2, 2)
#         sns.boxplot(x=x[col])
#         plt.title(f'{col} Boxplot (After Log Transform)')
#         plt.tight_layout()
#         plt.show()
#         print(f"{col} (After Log Transform) describe:\n", x[col].describe())
# else:
#     print("로그 변환이 적용된 컬럼이 없어 후 시각화를 진행하지 않습니다.")


# 2-2. 이상치(Outlier) 처리 (IQR 방식의 Winsorization)
def handle_outliers_iqr(df, column, train_q1=None, train_q3=None):
    if train_q1 is None or train_q3 is None: # train 데이터 처리 시
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
    else: # test 데이터 처리 시, train에서 계산된 Q1, Q3 사용
        Q1 = train_q1
        Q3 = train_q3
        
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 윈저라이징 적용 전에 이상치 개수 확인
    initial_outliers_lower = df[df[column] < lower_bound].shape[0]
    initial_outliers_upper = df[df[column] > upper_bound].shape[0]
    total_initial_outliers = initial_outliers_lower + initial_outliers_upper
    
    print(f"   컬럼 '{column}': 처리 전 이상치 개수 = {total_initial_outliers} (하한:{lower_bound:.2f}, 상한:{upper_bound:.2f})")

    # 하한보다 작은 값은 하한으로, 상한보다 큰 값은 상한으로 대체 (Winsorization)
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    return df, Q1, Q3 # train 처리 시 Q1, Q3 반환

outlier_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'] # 로그 변환된 Nodule_Size, TSH_Result 포함

# train 데이터에 이상치 처리 적용 및 Q1, Q3 값 저장
train_q_values = {}
print("\n##### Train 데이터 이상치 처리 시작 (로그 변환 후 데이터 기준) #####")
for col in outlier_cols:
    print(f"\n--- 컬럼: '{col}' ---")
    print(f"   '{col}' (처리 전) min: {x[col].min():.2f}, max: {x[col].max():.2f}")
    
    x, Q1, Q3 = handle_outliers_iqr(x, col)
    train_q_values[col] = {'Q1': Q1, 'Q3': Q3}
    
    print(f"   '{col}' (처리 후) min: {x[col].min():.2f}, max: {x[col].max():.2f}")

# test 데이터에 이상치 처리 적용 (train에서 계산된 Q1, Q3 사용)
print("\n##### Test 데이터 이상치 처리 시작 (로그 변환 후 데이터 기준, Train 기준) #####")
for col in outlier_cols:
    print(f"\n--- 컬럼: '{col}' ---")
    print(f"   '{col}' (처리 전) min: {test_csv_processed[col].min():.2f}, max: {test_csv_processed[col].max():.2f}")
    
    test_csv_processed, _, _ = handle_outliers_iqr(
        test_csv_processed, col,
        train_q1=train_q_values[col]['Q1'],
        train_q3=train_q_values[col]['Q3']
    )
    print(f"   '{col}' (처리 후) min: {test_csv_processed[col].min():.2f}, max: {test_csv_processed[col].max():.2f}")

# 2-3. 'Age' 컬럼 구간화 (Binning)
print("\n##### 'Age' 컬럼 구간화 (Binning) 적용 #####")
bins = [0, 20, 30, 40, 50, 60, 70, 80, x['Age'].max() + 1] # Age의 최대값보다 1 크게 설정
labels = ['~10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s~'] # 각 구간의 라벨

x['Age_Group'] = pd.cut(x['Age'], bins=bins, labels=labels, right=False)
test_csv_processed['Age_Group'] = pd.cut(test_csv_processed['Age'], bins=bins, labels=labels, right=False)

x = pd.get_dummies(x, columns=['Age_Group'], drop_first=True)
test_csv_processed = pd.get_dummies(test_csv_processed, columns=['Age_Group'], drop_first=True)

print("'Age_Group' 컬럼 생성 및 One-Hot Encoding 완료.")
print("x 데이터프레임의 'Age_Group' 컬럼 카테고리 분포:\n", x.filter(like='Age_Group_').sum())
print("\n")


# 2-4. 연속형 변수들 간의 다중공선성 (상관관계) 확인
print("\n##### 연속형 변수들 간의 상관관계 (다중공선성) 확인 #####")
correlation_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
corr_matrix = x[correlation_cols].corr()

# plt.figure(figsize=(8, 6)) # 필요시 주석 해제하여 확인
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Matrix of Continuous Variables')
# plt.show()

print("\n상관관계 매트릭스:\n", corr_matrix)

print("\n--- 2단계 확장 전처리 완료 ---")


# ================== 3단계: 특성 공학 (Feature Engineering) 시작 ==================
print("\n##### 3단계: 특성 공학 (Feature Engineering) 시작 #####")

# 작은 상수 (epsilon) 정의: 0으로 나누는 오류 방지
epsilon = 1e-6

# 아이디어 1: 갑상선 호르몬 비율 특성 (Ratio Features)
x['T4_TSH_Ratio'] = x['T4_Result'] / (x['TSH_Result'] + epsilon)
test_csv_processed['T4_TSH_Ratio'] = test_csv_processed['T4_Result'] / (test_csv_processed['TSH_Result'] + epsilon)
print("'T4_TSH_Ratio' 생성 완료.")

x['T3_T4_Ratio'] = x['T3_Result'] / (x['T4_Result'] + epsilon)
test_csv_processed['T3_T4_Ratio'] = test_csv_processed['T3_Result'] / (test_csv_processed['T4_Result'] + epsilon)
print("'T3_T4_Ratio' 생성 완료.")

# 아이디어 2: 특정 기준치 초과/미달 여부 (Binary Features)
# 로그 변환된 값의 평균으로 기준치를 잡는 것이 더 합리적일 수 있음
tsh_mean = x['TSH_Result'].mean()
t4_mean = x['T4_Result'].mean()
t3_mean = x['T3_Result'].mean()

x['TSH_High'] = (x['TSH_Result'] > tsh_mean).astype(int)
test_csv_processed['TSH_High'] = (test_csv_processed['TSH_Result'] > tsh_mean).astype(int)
print("'TSH_High' 생성 완료.")

x['T4_Low'] = (x['T4_Result'] < t4_mean).astype(int)
test_csv_processed['T4_Low'] = (test_csv_processed['T4_Result'] < t4_mean).astype(int)
print("'T4_Low' 생성 완료.")

x['T3_High'] = (x['T3_Result'] > t3_mean).astype(int)
test_csv_processed['T3_High'] = (test_csv_processed['T3_Result'] > t3_mean).astype(int)
print("'T3_High' 생성 완료.")


# 아이디어 3: 연령-결절 크기 상호작용 (Interaction Feature)
x['Age_Nodule_Interaction'] = x['Age'] * x['Nodule_Size']
test_csv_processed['Age_Nodule_Interaction'] = test_csv_processed['Age'] * test_csv_processed['Nodule_Size']
print("'Age_Nodule_Interaction' 생성 완료.")

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

# 스케일링
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_smote) # SMOTE가 적용된 데이터에 fit_transform
x_test_scaled = scaler.transform(x_test) # test 데이터는 transform만
test_csv_scaled = scaler.transform(test_csv_processed)

# SMOTE 적용 후 클래스 가중치 재계산 (기본 balanced)
class_weights_smote_base = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
class_weight_dict_smote = dict(enumerate(class_weights_smote_base))

# (★★★ 변경 사항: F1-Score 최적화를 위해 weight_multiplier 초기값을 1.0으로 설정 ★★★)
# 1.0은 'balanced'와 동일한 효과 (SMOTE 후 0.5/0.5). 필요시 1.5, 2.0 등으로 늘려가며 실험
weight_multiplier = 1.28 
class_weight_dict_smote[1] *= weight_multiplier 

print(f"\nClass weights (SMOTE 적용 후, 양성 클래스 가중치 조정): {class_weight_dict_smote}")


# ================== 4. 딥러닝 모델 ==================
model = Sequential()
model.add(Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1)) # (★★★ 변경 사항: 드롭아웃 비율 조정 - 과적합 완화 시도 ★★★)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) # (★★★ 변경 사항: 드롭아웃 비율 조정 ★★★)
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # (★★★ 변경 사항: 드롭아웃 비율 조정 ★★★)
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3)) # (★★★ 변경 사항: 드롭아웃 비율 조정 ★★★)
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

# (★★★ 변경 사항: patience 50으로 증가. val_loss가 0.9인 상태가 오래 지속되어 더 인내심을 가질 수도 있음 ★★★)
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True) 
model.fit(x_train_scaled, y_train_smote, # SMOTE가 적용된 데이터 사용
    epochs=500, # (★★★ 변경 사항: 에포크 증가. patience가 늘었으니 에포크도 증가 ★★★)
    batch_size=128, validation_split=0.2,
    callbacks=[es, mc_dl], class_weight=class_weight_dict_smote, verbose=2) # 조정된 클래스 가중치 사용

# 딥러닝 예측 확률 및 threshold 조정 (x_test_scaled 사용)
y_proba_dl = model.predict(x_test_scaled).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_dl)

# (★★★ 변경 사항: 딥러닝 모델 임계값을 F1-Score 최대화로 변경 ★★★)
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
# (★★★ 변경 사항: 부스팅 모델 가중치도 weight_multiplier 1.0으로 초기화 ★★★)
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
    # (★★★ 변경 사항: CatBoost의 class_weights를 조정된 weight_multiplier 기반으로 변경 ★★★)
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

# (★★★ 변경 사항: 앙상블 임계값을 F1-Score 최대화로 변경 ★★★)
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
# (★★★ 변경 사항: 제출 파일명에 F1-Score 최적화 전략 반영 ★★★)
submission_csv.to_csv(path + 'submission_f1_score_optimized3.csv') 
print("✅ submission_f1_score_optimized3.csv 파일 생성 완료!")