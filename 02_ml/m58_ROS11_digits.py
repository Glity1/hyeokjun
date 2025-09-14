import numpy as np
import pandas as pd
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import random
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 랜덤고정
seed = 123
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target    
feature_names = datasets.feature_names

le = LabelEncoder() 
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y,
)

ros = RandomOverSampler(random_state=seed,
             sampling_strategy='auto',  #default
            #  sampling_strategy=0.75,  # 최대값의 75% 지정
              #  sampling_strategy={0:10000, 1:10000},  # (array([0, 1, 2]), array([50, 53, 33], dtype=int64))  # 샘플데이터를 제한하지말고 전부다 엄청 늘려버리면 과적합이긴하지만 성능은 좋음.
            #   n_jobs=-1,  # 사용 가능한 모든 CPU 코어를 사용하라.   # imbleran 0.13버전 이상은  n_jobs는 안써도된다.
              )

x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64), array([146, 146, 146, 146, 146, 146, 146, 146, 146, 146], dtype=int64))
# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',  # ohe 안했음 
              optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

print(np.unique(y_train, return_counts=True))

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
# print(y_pred)
# print(y_pred.shape)

y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)
# print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('accuracy_score : ', acc)
print('f1_score : ', f1)


####################### 결과 ################################

#1. smote 사용 후 데이터 훈련.
# accuracy_score :  0.9805555555555555
# f1_score :  0.9803370590509541

# 재현's SMOTE
# accuracy_score :  1.0
# f1_score :  1.0
