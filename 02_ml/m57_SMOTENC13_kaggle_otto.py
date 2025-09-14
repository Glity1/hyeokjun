
#47-0 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
import time
import random
import matplotlib.pyplot as plt

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() 
y = le.fit_transform(y)

# print(train_csv.info())
# print(train_csv.head())

# categorical_indices = []
# for i, col in enumerate(x.columns):
#     if x[col].nunique() < 10:
#         categorical_indices.append(i)

# print("Categorical feature indices:", categorical_indices)

def detect_categorical_features(df, max_unique=10, print_result=True):
    categorical_cols = []
    
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= max_unique:
            categorical_cols.append(col)
    
    if print_result:
        print(f"✅ 추정된 범주형 컬럼 수: {len(categorical_cols)}개")
        print(f"▶️ 범주형 컬럼: {categorical_cols}")
    
    return categorical_cols

cat_cols = detect_categorical_features(x, max_unique=10)

cat_indices = [x.columns.get_loc(col) for col in cat_cols]
print("➡️ SMOTENC용 인덱스:", cat_indices)

# exit()
smotenc = SMOTENC(random_state=337, 
                  categorical_features= [5])
x_res, y_res = smotenc.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x_res, y_res, shuffle=True, random_state=123, train_size=0.8,  stratify=y_res
    )

#2. 모델
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)   
 
print("===========", model.__class__.__name__, "===========")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(" acc : ", acc)
f1 = f1_score(y_test, y_pred, average='macro')
print(" f1 : ", f1)

# =========== KNeighborsClassifier =========== ## SMOTE
#1. smote 사용 후 데이터 훈련.
# accuracy_score :  0.7969457013574661
# f1_score :  0.7659881827493988

# 재현's SMOTE
# accuracy_score :  0.7850678733031674
# f1_score :  0.7531828195407798


# =========== KNeighborsClassifier ===========  ## SMOTENC
# acc :  0.8697794624396967
# acc :  0.8697794624396967
# f1 :  0.8693385503360344