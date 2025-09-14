import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
import time
from sklearn. metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target  

y = np.rint(y).astype(int)
# print(y) #[0 1 2 ... 8 9 8]
# print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
)

scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis()
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

evr = lda.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print(np.cumsum(evr))

# value = [1.0, 0.9999, 0.9997]

# for i in value:
#     nums = np.argmax(evr_cumsum >= i) + 1                     # 언제부터 원하는 값이 시작되는가?
#     count = np.sum(evr_cumsum >= i)                         # 원하는 값의 개수는 총 몇인가?
#     print(f"{i:.7f} 이상: {nums}번째부터, 총 {count}개")

nums = [1, 5, 9]

for i in nums:
    print(f"\n====== LDA n_components = {i} ======")
    lda = LinearDiscriminantAnalysis(n_components=i)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)
    
    #2. 모델
    model = RandomForestClassifier(random_state=711)
    model.fit(x_train_lda, y_train)

    #4. 평가, 예측
    results = model.score(x_test_lda, y_test)  
    y_pred = model.predict(x_test_lda)           
    acc = accuracy_score(y_test, y_pred) 
    
    print(f" accuracy_score : {acc:.4f} ")
    
    
# ====== LDA n_components = 1 ======
#  accuracy_score : 0.3778 

# ====== LDA n_components = 5 ======
#  accuracy_score : 0.9278 

# ====== LDA n_components = 9 ======
#  accuracy_score : 0.9556 