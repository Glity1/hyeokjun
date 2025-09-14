import numpy as np 
from sklearn.datasets import load_iris

#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x,y = load_iris(return_X_y=True)

print(x.shape, y.shape) # (150,4) (150,)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential([
#     Dense(10, activation='relu', input_shape=(4,)),
#     Dense(10),
#     Dense(10),
#     Dense(3, activation = 'softmax')
# ])

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     # 유일하게 Regression이지만 분류모델
from sklearn.tree import DecisionTreeClassifier         # 분류 모델
from sklearn.ensemble import RandomForestClassifier     # 분류 모델

# model = LinearSVC(C=0.3)
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy',  #ohe안해도 가능하게 만듬.
#               optimizer='adam',
#               metrics=['acc'],
#               )
# model.fit(x, y, epochs=100)

model.fit(x, y)

#4. 평가, 예측
# results = model.evaluate(x,y,)
results = model.score(x,y)
print(results)
                # 0.96                   LinearSVC
                # 0.9733333333333334     LogisticRegression
                # 1.0                    DecisionTreeClassifier
                # 1.0                    RandomForestClassifier
                