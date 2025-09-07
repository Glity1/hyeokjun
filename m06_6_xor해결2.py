import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
np.random.seed(222)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = Perceptron()              # 안좋음  
# model = LinearSVC()               # 안좋음  
# model = SVC()                     # 잘된다
model = DecisionTreeClassifier()    # 잘된다

#3. 컴파일, 훈련 
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, np.round(y_predict))  
print("accuracy_score : ", acc)

# model.score :  1.0
# accuracy_score :  1.0








