# ================== 라이브러리 ==================
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # ModelCheckpoint 추가
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

# 시각화를 위한 라이브러리 추가
import matplotlib.pyplot as plt
import seaborn as sns

# 부스팅 모델
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier # CatBoost 모델은 개별적으로 사용하지만 Voting에서는 제외
# from sklearn.ensemble import VotingClassifier # VotingClassifier는 이제 사용하지 않음

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
bins = [0, 20, 30, 40, 50, 60, 70, 80, x['Age'].max() + 1] # Age의 최대값보다 1 크게 설정
labels = ['~10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s~'] # 각 구간의 라벨

x['Age_Group'] = pd.cut(x['Age'], bins=bins, labels=labels, right=False)
test_csv_processed['Age_Group'] = pd.cut(test_csv_processed['Age'], bins=bins, labels=labels, right=False)

x = pd.get_dummies(x, columns=['Age_Group'], drop_first=True)
test_csv_processed = pd.get_dummies(test_csv_processed, columns=['Age_Group'], drop_first=True)

# 원본 'Age' 컬럼은 유지 (또는 삭제하여 성능 비교 가능)
print("'Age_Group' 컬럼 생성 및 One-Hot Encoding 완료.")
print("x 데이터프레임의 'Age_Group' 컬럼 카테고리 분포:\n", x.filter(like='Age_Group_').sum())
print("\n")


# 2-4. 연속형 변수들 간의 다중공선성 (상관관계) 확인
print("\n##### 연속형 변수들 간의 상관관계 (다중공선성) 확인 #####")
correlation_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
corr_matrix = x[correlation_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Continuous Variables')
plt.show()

print("\n상관관계 매트릭스:\n", corr_matrix)

# 다중공선성 처리 (상관관계가 높은 경우) - 현재 데이터에서는 필요 없음
# if corr_matrix.loc['T4_Result', 'T3_Result'] > 0.8:
#    print("\n'T4_Result'와 'T3_Result' 간의 상관관계가 높으므로 'T3_Result'를 제거합니다.")
#    x = x.drop('T3_Result', axis=1)
#    test_csv_processed = test_csv_processed.drop('T3_Result', axis=1)

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
# 여기서는 각 호르몬의 평균을 기준으로 '높음'/'낮음'을 판단하는 예시를 듭니다.
# 실제로는 의학적 정상 범위를 사용하는 것이 더 좋습니다.
# 훈련 데이터의 평균을 기준으로 사용합니다.
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


# 아이디어 4: 다중 범주형 특성의 조합 (Combinations) - 선택사항 (컬럼 수 증가 유의)
# 예: 'Country'와 'Race'를 조합
# combined_df_for_combo = pd.concat([x.copy(), test_csv_processed.copy()], axis=0)
# combined_df_for_combo['Country_Race_Combined'] = combined_df_for_combo['Country'].astype(str) + '_' + combined_df_for_combo['Race'].astype(str)
# combined_df_for_combo = pd.get_dummies(combined_df_for_combo, columns=['Country_Race_Combined'], drop_first=True)
# x = combined_df_for_combo.iloc[:len(x)].copy()
# test_csv_processed = combined_df_for_combo.iloc[len(x):].copy()
# print("'Country_Race_Combined' 생성 및 One-Hot Encoding 완료.")

print("\n--- 3단계: 특성 공학 (Feature Engineering) 완료 ---")
print(f"최종 특성 개수 (x): {x.shape[1]}개")
print(f"최종 특성 개수 (test_csv_processed): {test_csv_processed.shape[1]}개")
print("이제 모델 학습을 진행합니다.")

# ================== 3단계: 특성 공학 (Feature Engineering) 끝 ==================


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

# ================== 4. 딥러닝 모델 ==================
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

checkpoint_filepath_dl = model_save_path + 'best_dl_model.h5'
mc_dl = ModelCheckpoint(
    filepath=checkpoint_filepath_dl,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False, # 전체 모델(구조+가중치) 저장
    verbose=1)

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

# ================== 5. 부스팅 모델 ==================
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

# ================== 6. 앙상블 (수동 Soft Voting) ==================
print("\n##### 앙상블 (수동 Soft Voting) 시작 - 모든 모델 포함 #####")

# 각 모델의 test_csv_scaled에 대한 예측 확률 얻기
# 딥러닝 모델은 predict()가 바로 확률을 반환
test_proba_dl = model.predict(test_csv_scaled).ravel()
# 부스팅 모델은 predict_proba()가 [클래스0 확률, 클래스1 확률]을 반환하므로 클래스1 확률만 사용
test_proba_xgb = xgb.predict_proba(test_csv_scaled)[:, 1]
test_proba_lgb = lgb_model.predict_proba(test_csv_scaled)[:, 1]
test_proba_cat = cat_model.predict_proba(test_csv_scaled)[:, 1]

# test_csv_scaled에 대한 모든 모델의 예측 확률 평균 (최종 제출 예측을 위함)
test_proba_ensemble = (test_proba_dl + test_proba_xgb + test_proba_lgb + test_proba_cat) / 4

# 각 모델의 x_test에 대한 예측 확률 얻기 (threshold 조정을 위함)
y_proba_dl_test = model.predict(x_test).ravel()
y_proba_xgb_test = xgb.predict_proba(x_test)[:, 1]
y_proba_lgb_test = lgb_model.predict_proba(x_test)[:, 1]
y_proba_cat_test = cat_model.predict_proba(x_test)[:, 1]


# 모든 모델의 예측 확률 평균 (Soft Voting)
# 이전에 딥러닝에서만 임계값을 찾았지만, 이제 앙상블 전체에 대한 임계값을 찾습니다.
# 모든 모델의 가중치를 동일하게 1/N으로 부여 (여기서는 4개 모델이므로 1/4)
y_proba_ensemble = (y_proba_dl_test + y_proba_xgb_test + y_proba_lgb_test + y_proba_cat_test) / 4 

# 앙상블 결과에 대한 F1 Score 기준 최적 임계값 찾기
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_ensemble)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh_ensemble = thresholds[best_idx]
y_pred_ensemble = (y_proba_ensemble >= best_thresh_ensemble).astype(int)

print(f"Ensemble Best Threshold: {best_thresh_ensemble}")
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble)}")
print(f"Ensemble F1 Score: {f1_score(y_test, y_pred_ensemble)}")

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
# ModelCheckpoint 콜백이 이미 학습 중에 최적의 모델을 저장했으므로, 여기서는 추가적인 저장은 필요 없음.
# 다만, 만약 EarlyStopping으로 restore_best_weights=True를 사용했다면,
# 학습이 끝난 모델 자체가 최적의 가중치를 가진 상태이므로 마지막 모델을 저장해도 무방합니다.
# 이미 mc_dl 콜백으로 best_dl_model.h5가 저장됨.
print(f"딥러닝 모델 (최적 가중치) 저장 완료: {checkpoint_filepath_dl}")

# 9-2. XGBoost 모델 저장
# XGBoost는 dump_model 또는 save_model로 모델 전체를 저장할 수 있습니다.
# JSON 형식으로 저장하는 것이 호환성 및 가독성 측면에서 더 좋습니다.
xgb_save_path = model_save_path + 'xgb_model.json'
xgb.save_model(xgb_save_path)
print(f"XGBoost 모델 저장 완료: {xgb_save_path}")

# 9-3. LightGBM 모델 저장
# LightGBM은 save_model 기능을 제공합니다.
import joblib
lgbm_save_path_joblib = model_save_path + 'lgbm_model.pkl' # .pkl 확장자 권장
joblib.dump(lgb_model, lgbm_save_path_joblib)
print(f"LightGBM 모델 저장 완료: {lgbm_save_path_joblib}")

# 9-4. CatBoost 모델 저장
# CatBoost는 save_model 기능을 제공합니다.
cat_save_path = model_save_path + 'catboost_model.cbm' # .cbm이 CatBoost의 권장 형식
cat_model.save_model(cat_save_path)
print(f"CatBoost 모델 저장 완료: {cat_save_path}")

print("모든 모델 및 가중치 저장 완료!")

# ================== 7. 제출 파일 생성 (수동 Soft Voting 기반) ==================
# test_csv_scaled에 대한 최종 예측은 위에서 계산된 best_thresh_ensemble 사용
final_test_pred = (test_proba_ensemble >= best_thresh_ensemble).astype(int)

submission_csv['Cancer'] = final_test_pred
path_1 = './_data/dacon/cancer/'
submission_csv.to_csv(path_1 + 'submission_2045_manual_ensemble.csv') # 파일명 변경
print("✅ submission_2045_manual_ensemble.csv 파일 생성 완료!")


### 암이 아닌 사람은 잘 찾으나 암 환자인 사람을 헷갈리거나 놓침 ###
