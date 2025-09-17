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


# 그래프 설정 (한글 깨짐 방지, 음수 부호 깨짐 방지 등)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준
plt.rcParams['axes.unicode_minus'] = False # 음수 부호 깨짐 방지
sns.set_style('whitegrid') # Seaborn 스타일 설정

# 데이터 로드 (경로는 당신의 실제 파일 위치에 맞게 수정해주세요)
path = './_data/dacon/cancer/' # 예시 경로, 실제 경로로 변경 필요
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print("--- 1. 데이터 로드 및 기본 정보 확인 ---")
print("Train 데이터 상위 5행:")
print(train_csv.head())
print("\nTest 데이터 상위 5행:")
print(test_csv.head())

print("\n--- 2. 데이터프레임 정보 (컬럼명, Non-Null Count, Dtype) ---")
print("\nTrain 데이터 Info:")
train_csv.info()
print("\nTest 데이터 Info:")
test_csv.info()

print("\n--- 3. 결측치 확인 ---")
print("\nTrain 데이터 결측치 수 (컬럼별):")
print(train_csv.isnull().sum()[train_csv.isnull().sum() > 0]) # 결측치가 있는 컬럼만 출력
print("\nTest 데이터 결측치 수 (컬럼별):")
print(test_csv.isnull().sum()[test_csv.isnull().sum() > 0]) # 결측치가 있는 컬럼만 출력

print("\n--- 4. 타겟 변수(Cancer) 분포 확인 ---")
print("Train 데이터 'Cancer' 클래스 분포:")
print(train_csv['Cancer'].value_counts())
print(f"클래스 1 비율: {train_csv['Cancer'].value_counts(normalize=True)[1]:.2f}")

print("\n--- 5. 주요 통계량 확인 (수치형 컬럼) ---")
print("\nTrain 데이터 수치형 컬럼 통계량:")
print(train_csv.describe())
print("\nTest 데이터 수치형 컬럼 통계량:")
print(test_csv.describe())

print("\n--- 6. 범주형 컬럼 고유값 수 확인 ---")
# 실제 범주형 컬럼명 리스트로 수정해주세요.
# 예시: ['Gender', 'Race', 'Country', 'Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes']
categorical_cols_to_check = ['Gender', 'Race', 'Country'] # 실제 컬럼명에 맞게 조정
for col in categorical_cols_to_check:
    if col in train_csv.columns:
        print(f"'{col}' 컬럼 고유값: {train_csv[col].unique()}")
        print(f"'{col}' 컬럼 고유값 수: {train_csv[col].nunique()}개")
        print(f"'{col}' 컬럼 값 분포:\n{train_csv[col].value_counts(dropna=False)}") # NaN도 포함하여 확인
        print("-" * 30)

print("\n--- 7. 연속형 컬럼 분포 시각화 (히스토그램 및 박스플롯) ---")
# 실제 연속형 컬럼명 리스트로 수정해주세요.
continuous_cols_to_check = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'] # 실제 컬럼명에 맞게 조정
for col in continuous_cols_to_check:
    if col in train_csv.columns:
        print(f"\n##### '{col}' 컬럼 시각화 #####")
        plt.figure(figsize=(16, 6))

        # 히스토그램
        plt.subplot(1, 2, 1)
        sns.histplot(train_csv[col], kde=True, bins=30)
        plt.title(f'{col} 분포 (Histogram & KDE)', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('빈도', fontsize=12)

        # 박스플롯 (이상치 확인)
        plt.subplot(1, 2, 2)
        sns.boxplot(x=train_csv[col])
        plt.title(f'{col} 이상치 (Boxplot)', fontsize=14)
        plt.xlabel(col, fontsize=12)
        
        plt.tight_layout()
        plt.show() # 각 그래프를 개별적으로 출력