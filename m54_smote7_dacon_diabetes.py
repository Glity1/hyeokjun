import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random
from imblearn.over_sampling import SMOTE

# 랜덤고정
seed = 123
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
feature_names = train_csv.columns

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

print(x.shape, y.shape) #(652, 8) (652,)
print(np.unique(y, return_counts=True)) # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))
print(pd.value_counts(y))
# 0    424
# 1    228
# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, shuffle=True,
    stratify=y
    )


smote = SMOTE(random_state=seed,
              k_neighbors=5,
            #  sampling_strategy='auto',  #default
            #  sampling_strategy=0.75,  # 최대값의 75% 지정
              # sampling_strategy={0:424, 1:424},  # (array([0, 1, 2]), array([50, 53, 33], dtype=int64))  # 샘플데이터를 제한하지말고 전부다 엄청 늘려버리면 과적합이긴하지만 성능은 좋음.
              sampling_strategy={0:10000, 1:100000},
              n_jobs=-1,  # 사용 가능한 모든 CPU 코어를 사용하라.   # imbleran 0.13버전 이상은  n_jobs는 안써도된다.
              )

x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))
#(array([0, 1], dtype=int64), array([424, 424], dtype=int64))

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',  # ohe 안했음 
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

y_pred = (y_pred > 0.5).astype(int)
# print(y_pred)
# print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('accuracy_score : ', acc)
print('f1_score : ', f1)


####################### 결과 ################################

#1. smote 사용 후 데이터 훈련.
# accuracy_score :  0.6625766871165644
# f1_score :  0.6275291869209356

# 재현's SMOTE
# accuracy_score :  0.5705521472392638
# f1_score :  0.556514303482587
