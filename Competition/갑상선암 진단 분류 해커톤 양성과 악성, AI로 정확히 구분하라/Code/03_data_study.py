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
x = combined_df.iloc[:len(train_csv)].copy()
test_csv_processed = combined_df.iloc[len(train_csv):].copy()

# 'Cancer' 타겟 변수는 그대로 사용
y = train_csv['Cancer']

# --- 2단계: 연속형 변수 전처리 적용 시작 ---

# 2-1. 분포 변환 (로그 변환) 대상 컬럼 정의
# 모든 연속형 변수가 균등 분포를 보였으므로, 로그 변환 대상에서 모두 제외합니다.
log_transform_cols = [] # 빈 리스트로 만듭니다.

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

# 실제 로그 변환 적용 (log_transform_cols가 비어 있으므로, 이 반복문은 아무것도 실행하지 않습니다.)
for col in log_transform_cols:
    x[col] = np.log1p(x[col])
    test_csv_processed[col] = np.log1p(test_csv_processed[col])
    print(f"'{col}' 컬럼에 로그 변환 적용 완료.")

# --- 로그 변환 후 시각화 및 통계량 (로그 변환 적용된 컬럼이 없으므로, 이 부분은 아무것도 그리지 않습니다.) ---
print("\n##### 로그 변환 후 (적용된 컬럼) 분포 시각화 #####")
if log_transform_cols: # log_transform_cols가 비어있지 않을 때만 실행
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
else:
    print("로그 변환이 적용된 컬럼이 없어 후 시각화를 진행하지 않습니다.")


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
    
    return df, Q1, Q3 # train 처리 시 Q1, Q3 반환

outlier_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'] # 시각화 결과 이상치 없음, 그대로 유지

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


# --- 2단계 추가: 연속형 변수 추가 전처리 ---

# 2-3. 'Age' 컬럼 구간화 (Binning)
print("\n##### 'Age' 컬럼 구간화 (Binning) 적용 #####")
# 구간을 정의합니다. (예: 10대, 20대, 30대... 80대 이상)
# 데이터의 최소/최대 나이를 고려하여 적절히 조절할 수 있습니다.
# 예를 들어, 10세 미만은 0, 10-19세는 1, 20-29세는 2 등으로 숫자로 라벨링 할 수도 있습니다.
bins = [0, 20, 30, 40, 50, 60, 70, 80, x['Age'].max() + 1] # Age의 최대값보다 1 크게 설정
labels = ['~10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s~'] # 각 구간의 라벨

x['Age_Group'] = pd.cut(x['Age'], bins=bins, labels=labels, right=False) # right=False는 [a,b) 구간 (a이상 b미만)
test_csv_processed['Age_Group'] = pd.cut(test_csv_processed['Age'], bins=bins, labels=labels, right=False)

# 새로 생성된 'Age_Group' 컬럼을 One-Hot Encoding (범주형으로 변환되었으므로)
# drop_first=True로 첫 번째 카테고리 제거 (다중공선성 방지)
x = pd.get_dummies(x, columns=['Age_Group'], drop_first=True)
test_csv_processed = pd.get_dummies(test_csv_processed, columns=['Age_Group'], drop_first=True)

# 원본 'Age' 컬럼은 더 이상 사용하지 않을 경우 삭제 (또는 유지하여 성능 비교)
# 여기서는 원본 Age 컬럼을 유지하여 모델이 두 가지 형태의 나이 정보를 모두 활용하도록 합니다.
# 만약 'Age'를 삭제하려면 아래 주석을 해제하세요.
# x = x.drop('Age', axis=1)
# test_csv_processed = test_csv_processed.drop('Age', axis=1)

print("'Age_Group' 컬럼 생성 및 One-Hot Encoding 완료.")
print("x 데이터프레임의 'Age_Group' 컬럼 카테고리 분포:\n", x.filter(like='Age_Group_').sum())
print("\n")


# 2-4. 연속형 변수들 간의 다중공선성 (상관관계) 확인
print("\n##### 연속형 변수들 간의 상관관계 (다중공선성) 확인 #####")
# Age_Group은 원-핫 인코딩 되었으므로 상관관계 확인 대상에서 제외하고 원본 Age 포함
correlation_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
corr_matrix = x[correlation_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Continuous Variables')
plt.show()

print("\n상관관계 매트릭스:\n", corr_matrix)

# 다중공선성 처리 (상관관계가 높은 경우)
# 현재 데이터셋에서는 높은 상관관계가 없을 것으로 예상되지만, 확인 후 판단.
# 예시: 만약 'T4_Result'와 'T3_Result'의 상관관계가 0.8 이상으로 매우 높다면,
# 둘 중 하나를 제거하는 것을 고려할 수 있습니다.
# if corr_matrix.loc['T4_Result', 'T3_Result'] > 0.8:
#    print("\n'T4_Result'와 'T3_Result' 간의 상관관계가 높으므로 'T3_Result'를 제거합니다.")
#    x = x.drop('T3_Result', axis=1)
#    test_csv_processed = test_csv_processed.drop('T3_Result', axis=1)

print("\n--- 2단계 확장 전처리 완료 ---")


# --- 2단계: 연속형 변수 전처리 적용 끝 ---


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=777, shuffle=True, stratify=y
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
model.fit(x_train, y_train,
 epochs=400, batch_size=128, validation_split=0.1,
 callbacks=[es], class_weight=class_weight_dict, verbose=2)

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
 iterations=500, learning_rate=0.05, depth=6, verbose=0,
 random_state=42, class_weights=[class_weight_dict[0], class_weight_dict[1]]
)
cat_model.fit(x_train, y_train)
y_pred_cat = cat_model.predict(x_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost F1 Score:", f1_score(y_test, y_pred_cat))

# ================== 4. 앙상블 (VotingClassifier) ==================
voting = VotingClassifier(
 estimators=[
 ('xgb', xgb),
 ('lgb', lgb_model),
 ('cat', cat_model),
],
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