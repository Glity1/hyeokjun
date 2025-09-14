import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() # 문자열을 숫자로 현재 y에 들어간 train_csv의 target은 문자열임
y = le.fit_transform(y)  

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)                   # StratifiedKFold 범주형일 때 확실하게 분류 // 데이터가 많으면 kfold써도됨.

#2. 모델구성
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

# acc :  [0.78012322 0.79173821 0.78252525 0.7769697  0.78505051] 
#  평균 acc :  0.7833

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
y_pred = np.round(y_pred)
                 
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc) # cross_val_predict ACC :   0.7483031674208145
