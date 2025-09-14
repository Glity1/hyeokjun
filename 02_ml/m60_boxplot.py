#47_00 copy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score
import time
import random
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target

df.boxplot()
# plt.show()

# 이상치 비율을 계산하기 위한 빈 딕셔너리
outlier_counts = {}

# IQR 방식으로 이상치 개수 계산
for col in df.columns[:-1]:  # 'target' 제외
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # 이상치 개수
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_counts[col] = len(outliers)

# 이상치 개수가 많은 순으로 정렬
sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)

# 이상치가 0개 초과인 컬럼만 출력
print("📌 이상치가 많은 컬럼:")
for col, count in sorted_outliers:
    if count > 0:
        print(f"{col}: {count}개")

x = datasets.data
y = datasets.target

# log 변환할 컬럼 수동 지정
log_cols = ['AveBedrms', 'Population', 'AveOccup', 'MedInc', 'AveRooms']

# log1p 변환 적용
for col in log_cols:
    df[col] = np.log1p(df[col])

log_x = np.log1p(x)
log_y = np.log1p(y)

# 결과 저장용
results = {}

# 1. 기본 (x, y 원본)
x1, y1 = x, y
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["log 변환 전 score : "] = r2_score(y_test, y_pred)

# 2. y만 log
x2, y2 = x, log_y
x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)         # 역변환
y_true = np.expm1(y_test)             # 평가 대상도 역변환해야 함
results["y만 log 변환 score : "] = r2_score(y_true, y_pred)

# 3. x만 log
x3, y3 = log_x, y
x_train, x_test, y_train, y_test = train_test_split(x3, y3, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["x만 log 변환 score : "] = r2_score(y_test, y_pred)

# 4. x, y 모두 log
x4, y4 = log_x, log_y
x_train, x_test, y_train, y_test = train_test_split(x4, y4, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
results["x,y log 변환 score : "] = r2_score(y_true, y_pred)

# 5. x 일부만 log 변환
x_log_partial = df[datasets.feature_names].values
x5, y5 = x_log_partial, y
x_train, x_test, y_train, y_test = train_test_split(x5, y5, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["x 이상치 많은 컬럼만 log 변환 score : "] = r2_score(y_test, y_pred)


# 결과 출력
for k, v in results.items():
    print(f"{k} R2 Score: {v:.4f}")

# RandomForestRegressor 모델로
# log 변환 전 score :  R2 Score: 0.8122
# y만 log 변환 score :  R2 Score: 0.8142
# x만 log 변환 score :  R2 Score: 0.7327
# x,y log 변환 score :  R2 Score: 0.7364
# x 이상치 많은 컬럼만 log 변환 score : R2 Score: 0.8123