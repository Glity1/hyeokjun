import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes                               #당뇨병 diabetes
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
import time
from sklearn. metrics import accuracy_score, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape) # (442, 10)
print(y) # 수치형 데이터 (부동소수점) 

y_origin = y.copy()

y = np.rint(y).astype(int) #-> int형으로 바꿔야함
# print(y)  # 정수형 데이터로 변경됨 -> 범주형으로 이제 가능
# print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test, y_train_o, y_test_o = train_test_split(
    x, y, y_origin, test_size=0.2, random_state=222,
)

scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# lda = LinearDiscriminantAnalysis()
# x_train = lda.fit_transform(x_train, y_train)
# x_test = lda.transform(x_test)

# evr = lda.explained_variance_ratio_
# evr_cumsum = np.cumsum(evr)

# print(evr_cumsum)
# print(np.cumsum(evr))

# value = [0.9900, 0.8300, 0.4000]

# for i in value:
#     nums = np.argmax(evr_cumsum >= i) + 1                     # 언제부터 원하는 값이 시작되는가?
#     count = np.sum(evr_cumsum >= i)                         # 원하는 값의 개수는 총 몇인가?
#     print(f"{i:.7f} 이상: {nums}번째부터, 총 {count}개")

# # 0.9900000 이상: 8번째부터, 총 3개
# # 0.8300000 이상: 5번째부터, 총 6개
# # 0.4000000 이상: 1번째부터, 총 10개

# # exit()

nums = [1, 5, 8]

for i in nums:
    print(f"\n====== LDA n_components = {i} ======")
    lda = LinearDiscriminantAnalysis(n_components=i)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)
    
    #2. 모델
    model = RandomForestRegressor(random_state=711)
    model.fit(x_train_lda, y_train_o)

    #4. 평가, 예측
    results = model.score(x_test_lda, y_test_o)  
    y_pred = model.predict(x_test_lda)           
    r2 = r2_score(y_test_o, y_pred) 
    
    print(f" R2_score : {r2:.4f} ")