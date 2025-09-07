import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.DESCR)

print(datasets.feature_names) 
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 
# 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 
# 'od280/od315_of_diluted_wines', 'proline']

print(type(datasets))
#<class 'sklearn.utils._bunch.Bunch'>


print(x.shape, y.shape)                  #(178, 13) (178,)
print(np.unique(y, return_counts=True))  #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y))
#1    71
#0    59
#2    48
#dtype: int64

y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=814,
    shuffle=True, 
)

print(x_train.shape, x_test.shape)  #(142, 13) (36, 13)
print(y_train.shape, y_test.shape)  #(142, 3) (36, 3)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# exit()

x_train = x_train.reshape(142,13,1,1)
x_test = x_test.reshape(36,13,1,1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(13, 1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=3, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   patience=30, 
                   restore_best_weights=True)



hist = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=256,
    validation_split=0.2,
    callbacks=[es],
    verbose=2
)


#4. 평가, 예측
y_predict = model.evaluate(x_test, y_test)
print("loss : ", y_predict[0])
print('acc : ', y_predict[1])       

"""
loss :  0.9249485731124878
acc :  0.3055555522441864

dropout 변경 후

loss :  0.004574982449412346
acc :  1.0

"""
