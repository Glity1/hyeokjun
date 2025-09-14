import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import LabelEncoder
import sklearn as sk
print(sk.__version__)
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
    x, y, shuffle=True, random_state=333, test_size=0.2,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = RandomForestRegreeor()
allAlgorithms = all_estimators(type_filter='classifier')

# print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms)) #모델의 갯수 :  55
print(type(allAlgorithms)) #<class 'list'>

max_acc = 0
max_name = '최고'

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, '의 정답률 : ', results)    
       
        if results > max_acc:
           max_acc = results
           max_name = name        
        
    except:
        print(name, "은(는) 정답률을 생성할 수 없습니다!")
        
print("==============================================================================================")
print("최고모델 : ", max_name, max_acc)
print("==============================================================================================")

# 최고모델 :  HistGradientBoostingClassifier 0.8652710031205502



