import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

le = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited'] 

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

# acc :  [0.41332842 0.40668953 0.40876932 0.40867425 0.40761538] 
#  평균 acc :  0.409

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
                 
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc) #cross_val_predict ACC :  0.19172978421572695


