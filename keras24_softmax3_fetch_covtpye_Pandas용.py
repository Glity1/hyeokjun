import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

y = pd.get_dummies(y)
print(y)
print(y.shape) #(581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=814,
    shuffle=True, stratify=y .values.argmax(axis=1) # 0,1,2 를 균등하게
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   patience=30, 
                   restore_best_weights=True)

hist = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=2048,
    validation_split=0.2,
    callbacks=[es],
    verbose=2
)
#4. 평가, 예측
y_predict = model.evaluate(x_test, y_test)
print("loss : ", y_predict[0])
print('acc : ', y_predict[1])   
# f1 = f1_score(y_test, y_predict) 
# print("f1_score : ", f1)

# loss :  0.2608710825443268
# acc :  0.8963624238967896