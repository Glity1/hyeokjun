# https://dacon.io/competitions/official/236068/leaderboard

import numpy as np                                           
import pandas as pd                                          
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
path = './_data/dacon/diabetes/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
print(train_csv)                    # [652 rows x 9 columns]             

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [116 rows x 1 columns] 

print(train_csv.shape)              #(652,9)
print(test_csv.shape)               #(116,8)
print(submission_csv.shape)         #(116,1)

print("######################################################################################################################")                                              
print(train_csv.columns)  # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
                          # dtype='object')
print("######################################################################################################################")                                              
print(train_csv.info())             
print("######################################################################################################################")                          
print(train_csv.describe())

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome'] 
x = x.replace(0, np.nan)                
x = x.fillna(x.mean())
test_csv = test_csv.fillna(x.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())

print(x) #652 rows x 8 
print(y) #652

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True,
    )
print(x_train.shape, x_test.shape)  #(521, 8) (131, 8)
print(y_train.shape, y_test.shape)  #(521,)  (131,)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# exit()

x_train = x_train.reshape(521,2,2,2)
x_test = x_test.reshape(131,2,2,2)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(2, 2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics=['acc'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience=30,
    restore_best_weights=True,
)


hist = model.fit(x_train, y_train, epochs=500, batch_size=4,
          verbose = 1,
          validation_split= 0.2,
          callbacks=[es],
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)                                 # binary_crossentropy의 loss 와 accuracy를 볼 수 있다.
# [0.011887827888131142, 1.0]
# exit()

print("loss : ", results[0])                   # loss : 0.022432664409279823
print("accuracy : ",round(results[1], 5))      # acc  : 0.987246

y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict)                            # python 그냥 씀 / numpy 는 np.
# print(y_predict[:10])

print(y_test.shape, y_predict.shape) #(131,) (131, 1)
# y_test = np.array(y_test).reshape(131,1)

print(y_predict.shape) #(66,1)
from sklearn.metrics import accuracy_score
accuracy_score= accuracy_score(y_test, y_predict)     # 변수 = 함수()
print("acc_score : ", accuracy_score)

# loss :  0.4718417525291443
# accuracy :  0.35115
# (131,) (131, 1)
# (131, 1)
# acc_score :  0.3511450381679389
