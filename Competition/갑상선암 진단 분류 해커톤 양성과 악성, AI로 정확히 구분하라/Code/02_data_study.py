# ================== 라이브러리 ==================
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# 시각화를 위한 라이브러리 추가
import matplotlib.pyplot as plt
import seaborn as sns

# 부스팅 모델
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# ================== 1. 데이터 로딩 및 전처리 ==================
path = './_data/dacon/cancer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

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
x = combined_df.iloc[:len(train_csv)].copy() # .copy()를 사용하여 SettingWithCopyWarning 방지
test_csv_processed = combined_df.iloc[len(train_csv):].copy() # .copy() 사용

# 'Cancer' 타겟 변수는 그대로 사용
y = train_csv['Cancer']

# --- 2단계: 연속형 변수 전처리 적용 시작 ---

# 2-1. 분포 변환 (로그 변환) 대상 컬럼 정의
# Nodule_Size는 이전 분석 결과에 따라 필요 없을 수 있으므로 일단 제외하고 시작
log_transform_cols = ['TSH_Result', 'T4_Result', 'T3_Result']
# 'Age'는 로그 변환보다 다른 처리(구간화 등)가 더 적합할 수 있지만, 일단 확인 목적에는 포함하지 않음

# --- 로그 변환 전 시각화 및 통계량 ---
print("\n##### 로그 변환 전 각 컬럼의 분포 및 이상치 시각화 #####")
for col in continuous_cols_initial: # 모든 연속형 컬럼에 대해 일단 확인
    print(f"\n--- 컬럼: '{col}' (로그 변환 전) ---")
    plt.figure(figsize=(14, 5))
    
    # 히스토그램
    plt.subplot(1, 2, 1)
    sns.histplot(x[col], kde=True)
    plt.title(f'{col} (Before Log Transform)')
    
    # 박스플롯
    plt.subplot(1, 2, 2)
    sns.boxplot(x=x[col])
    plt.title(f'{col} Boxplot (Before Log Transform)')
    
    plt.tight_layout()
    plt.show()
    print(f"{col} (Before Log Transform) describe:\n", x[col].describe())

# 실제 로그 변환 적용 (Nodule_Size는 분석 결과에 따라 제외)
for col in log_transform_cols:
    x[col] = np.log1p(x[col])
    test_csv_processed[col] = np.log1p(test_csv_processed[col])
    print(f"'{col}' 컬럼에 로그 변환 적용 완료.")

# --- 로그 변환 후 시각화 및 통계량 (로그 변환 적용된 컬럼만) ---
print("\n##### 로그 변환 후 (적용된 컬럼) 분포 시각화 #####")
for col in log_transform_cols:
    print(f"\n--- 컬럼: '{col}' (로그 변환 후) ---")
    plt.figure(figsize=(14, 5))
    
    # 히스토그램
    plt.subplot(1, 2, 1)
    sns.histplot(x[col], kde=True)
    plt.title(f'{col} (After Log Transform)')
    
    # 박스플롯
    plt.subplot(1, 2, 2)
    sns.boxplot(x=x[col])
    plt.title(f'{col} Boxplot (After Log Transform)')
    
    plt.tight_layout()
    plt.show()
    print(f"{col} (After Log Transform) describe:\n", x[col].describe())


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
    
    print(f"  컬럼 '{column}': 처리 전 이상치 개수 = {total_initial_outliers} (하한:{lower_bound:.2f}, 상한:{upper_bound:.2f})")

    # 하한보다 작은 값은 하한으로, 상한보다 큰 값은 상한으로 대체 (Winsorization)
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    # 윈저라이징 적용 후 이상치 개수 확인 (이제 0개여야 함)
    # final_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    # print(f"  컬럼 '{column}': 처리 후 이상치 개수 = {final_outliers}") # 항상 0일 것이므로 생략 가능
    
    return df, Q1, Q3 # train 처리 시 Q1, Q3 반환

# 'Nodule_Size'는 이전 분석 결과에 따라 이상치 처리 대상에서 제외할지 고려
# 여기서는 일단 모두 포함시켜서 결과를 보고 판단할 수 있도록 남겨둡니다.
outlier_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']


# train 데이터에 이상치 처리 적용 및 Q1, Q3 값 저장
train_q_values = {}
print("\n##### Train 데이터 이상치 처리 시작 (로그 변환 후 데이터 기준) #####")
for col in outlier_cols:
    print(f"\n--- 컬럼: '{col}' ---")
    # 처리 전 통계량
    print(f"  '{col}' (처리 전) min: {x[col].min():.2f}, max: {x[col].max():.2f}")
    
    # 처리 전 박스플롯 (로그 변환 후, 이상치 처리 전)
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=x[col])
    plt.title(f'{col} Boxplot (After Log, Before Winsorization)')
    plt.show()

    x, Q1, Q3 = handle_outliers_iqr(x, col)
    train_q_values[col] = {'Q1': Q1, 'Q3': Q3}
    
    # 처리 후 통계량
    print(f"  '{col}' (처리 후) min: {x[col].min():.2f}, max: {x[col].max():.2f}")
    
    # 처리 후 박스플롯
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=x[col])
    plt.title(f'{col} Boxplot (After Log, After Winsorization)')
    plt.show()


# test 데이터에 이상치 처리 적용 (train에서 계산된 Q1, Q3 사용)
print("\n##### Test 데이터 이상치 처리 시작 (로그 변환 후 데이터 기준, Train 기준) #####")
for col in outlier_cols:
    print(f"\n--- 컬럼: '{col}' ---")
    print(f"  '{col}' (처리 전) min: {test_csv_processed[col].min():.2f}, max: {test_csv_processed[col].max():.2f}")
    
    test_csv_processed, _, _ = handle_outliers_iqr(
        test_csv_processed, col,
        train_q1=train_q_values[col]['Q1'],
        train_q3=train_q_values[col]['Q3']
    )
    print(f"  '{col}' (처리 후) min: {test_csv_processed[col].min():.2f}, max: {test_csv_processed[col].max():.2f}")

# --- 2단계: 연속형 변수 전처리 적용 끝 ---


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777, shuffle=True, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv_processed)

# 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ================== 2. 딥러닝 모델 ==================
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=40, restore_best_weights=True)
model.fit(x_train, y_train,epochs=400, batch_size=128, validation_split=0.1, callbacks=[es], class_weight=class_weight_dict, verbose=2)

# 딥러닝 예측 확률 및 threshold 조정
y_proba_dl = model.predict(x_test).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_dl)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh_dl = thresholds[best_idx]
y_pred_dl = (y_proba_dl >= best_thresh_dl).astype(int)

print(f"DL Best Threshold: {best_thresh_dl}")
print(f"DL Accuracy: {accuracy_score(y_test, y_pred_dl)}")
print(f"DL F1 Score: {f1_score(y_test, y_pred_dl)}")

# ================== 3. 부스팅 모델 ==================
xgb = XGBClassifier(
n_estimators=500, learning_rate=0.05, max_depth=6,
subsample=0.8, colsample_bytree=0.8,
random_state=42,
scale_pos_weight=class_weight_dict[0] / class_weight_dict[1]
)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGB F1 Score:", f1_score(y_test, y_pred_xgb))

lgb_model = lgb.LGBMClassifier(
n_estimators=500, learning_rate=0.05, max_depth=6,
class_weight='balanced', random_state=42
)
lgb_model.fit(x_train, y_train)
y_pred_lgb = lgb_model.predict(x_test)
print("LGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LGBM F1 Score:", f1_score(y_test, y_pred_lgb))

cat_model = CatBoostClassifier(
iterations=500, learning_rate=0.05, depth=6, verbose=0,random_state=42, class_weights=[class_weight_dict[0], class_weight_dict[1]]
)
cat_model.fit(x_train, y_train)
y_pred_cat = cat_model.predict(x_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost F1 Score:", f1_score(y_test, y_pred_cat))

# ================== 4. 앙상블 (VotingClassifier) ==================
voting = VotingClassifier(estimators=[('xgb', xgb), ('lgb', lgb_model), ('cat', cat_model), ],
voting='soft'
)
voting.fit(x_train, y_train)
y_pred_vote = voting.predict(x_test)
print("Voting Accuracy:", accuracy_score(y_test, y_pred_vote))
print("Voting F1 Score:", f1_score(y_test, y_pred_vote))

# ================== 5. 제출 파일 생성 (Voting 기반) ==================
test_pred = voting.predict(test_csv_scaled)
submission_csv['Cancer'] = test_pred
path_1 = './_data/dacon/cancer/'
submission_csv.to_csv(path_1 + 'submission_2045.csv')
print("✅ submission_1808.csv 파일 생성 완료!")