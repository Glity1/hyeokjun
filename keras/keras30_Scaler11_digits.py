import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(datasets.DESCR)

print(datasets.feature_names)               # ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 
                                            # 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']


print(type(datasets))
                                            # <class 'sklearn.utils.Bunch'>


print(x.shape, y.shape)                     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))  



y = to_categorical(y) 
print(y) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# [[1. 0. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 1. 0.]
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 1. 0.]]

print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=814,
    shuffle=True, 
)

print(x_train.shape, y_train.shape) #(1437, 64) (1437, 10)
print(x_test.shape, y_test.shape)  #(360, 64) (360, 10)

x_train = x_train.reshape(1437,8,8)

print(y_train[0])

plt.imshow(x_train[0],'gray')
plt.show()

exit()

print(np.unique(y_train, return_counts=True)) # (array([0., 1.], dtype=float32), array([12933,  1437], dtype=int64))
print(np.unique(y_test, return_counts=True))  # (array([0., 1.], dtype=float32), array([3240,  360], dtype=int64))

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64) 
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)



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

path='./_save/keras30/Scaler11_digits/'
model.save_weights(path+'keras30_Scaler11_digits_weights.h5')

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   patience=30, 
                   restore_best_weights=True)

path='./_save/keras30/Scaler11_digits/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras30_Scaler11_digits.hdf5'
    )

hist = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es,mcp],
    verbose=2
)

path='./_save/keras30/Scaler11_digits/'
model.save(path+'keras30_Scaler11_digits.h5')

#4. 평가, 예측
y_predict = model.evaluate(x_test, y_test)
print("loss : ", y_predict[0])
print('acc : ', y_predict[1])       

loss :  0.0711914449930191
acc :  0.9750000238418579
