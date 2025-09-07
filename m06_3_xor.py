import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression     
from sklearn.tree import DecisionTreeClassifier         
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
      
# model_list = [
#     Perceptron(), 
#     LinearSVC(C=0.3),
# ]

# for model in model_list:
#     model_name = model.__class__.__name__
#     model.fit(x_data, y_data)
#     y_predict = model.predict(x_data)
#     results = model.score(x_data, y_data)
#     print("model.score : ", results)

#     acc = accuracy_score(y_data, y_predict)
#     print("accuracy_score : ", acc) 

#2. 모델
# model = Perceptron()      
model = LinearSVC()       

#3. 컴파일, 훈련 
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc)


# model.score :  0.5
# accuracy_score :  0.5 