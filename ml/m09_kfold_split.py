import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

#1. 데이터
datasets = load_iris()      
x = datasets.data
y = datasets['target']

# print(x) #[[5.1 3.5 1.4 0.2] ...  [5.9 3.  5.1 1.8]]
# print(y) # 0~2까지 순서대로 들어가있음
# print(type(x))
# print(x.shape)  #(150, 4)
# print(y.shape)  #(150)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True)                        

# kfold 확인

# for train_index, val_index in kfold.split(df):                                      # split 은 두개의 인덱스 리스트를 반환 
#     print(f"========================================================")
#     print(train_index, '\n', val_index)


for line_number, (train_index, val_index) in enumerate(kfold.split(df), start=1):          # split 은 두개의 인덱스 리스트를 반환 
    print(f"============================[{line_number}]============================")
    print(train_index, '\n', val_index)


# 선생님 코드
# for idx, (train_index, val_index) in enumerate(kfold.split(df)):                                      # split 은 두개의 인덱스 리스트를 반환 
#     print("============================[", idx,"]============================")
#     print(train_index, '\n', val_index)



