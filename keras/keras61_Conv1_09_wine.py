import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import BatchNormalization
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

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (142, 13) (36, 13) 
print(y_train.shape, y_test.shape) # (142,) (36,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(13, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   patience=30, 
                   restore_best_weights=True)

path='./_save/keras28_mcp/09_wine/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras33_mcp_09_wine.hdf5'
    )

hist = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=256,
    validation_split=0.2,
    callbacks=[es,mcp],
    verbose=2
)

path='./_save/keras28_mcp/09_wine/'
model.save(path+'keras33_mcp_save_09_wine.h5')

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

###
loss :  1.0972188711166382
acc :  0.2777777910232544

"""
