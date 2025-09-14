import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression     
from sklearn.tree import DecisionTreeClassifier         
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델

# model = Perceptron()      
# model = LinearSVC()
model = Sequential([
    Dense(1, input_dim=2, activation='sigmoid')
])       

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=10000)

#4. 평가, 예측
y_predict = model.predict(x_data)
# results = model.score(x_data, y_data)
results = model.evaluate(x_data, y_data)
print("model.evaluate : ", results)

acc = accuracy_score(y_data, np.round(y_predict))   # accuracy_score() 함수에 넣은 y_true(정답)와 y_pred(예측값)의 타입이 서로 다르다.
                                                    # 하나는 **이진 클래스(0, 1)**인데,
                                                    # 다른 하나는 **실수값(0.8, 0.3 같은 확률)**이 들어가서 에러가 난 거야.
                                                    # accuracy_score() 함수는 입력된 두 배열(y_data, y_predict)이 아래 조건을 만족해야 한다:
                                                    # 둘 다 같은 길이여야 하고,
                                                    # 둘 다 클래스(label) 형태여야 함 (예: 0, 1, 2 같은 정수형 클래스)

print("accuracy_score : ", acc)


# model.score :  0.5
# accuracy_score :  0.5 