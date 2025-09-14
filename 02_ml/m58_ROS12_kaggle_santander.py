import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import random
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tensorflow.python.keras.callbacks import EarlyStopping

# 랜덤고정
seed = 123
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
path = './_data/kaggle/santander/'                                          
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)                      

x = train_csv.drop(['target'], axis=1)                                     
y = train_csv['target']                                                    

feature_names = list(x.columns)

print(x.shape, y.shape) 
print(np.unique(y, return_counts=True)) 
print(pd.value_counts(y))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, shuffle=True,
    stratify=y
    )

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ros = RandomOverSampler(random_state=seed,
             sampling_strategy='auto',  #default
            #  sampling_strategy=0.75,  # 최대값의 75% 지정
              #  sampling_strategy={0:10000, 1:10000},  # (array([0, 1, 2]), array([50, 53, 33], dtype=int64))  # 샘플데이터를 제한하지말고 전부다 엄청 늘려버리면 과적합이긴하지만 성능은 좋음.
            #   n_jobs=-1,  # 사용 가능한 모든 CPU 코어를 사용하라.   # imbleran 0.13버전 이상은  n_jobs는 안써도된다.
              )

x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',  # ohe 안했음 
              optimizer='adam', metrics=['acc'])

es = EarlyStopping(                  
    monitor='val_loss',
    mode = 'min',                      
    patience=10,                       
    restore_best_weights=True,        
) 

model.fit(x,y, epochs=100, batch_size=128,
          verbose =2,
          validation_split=0.2,
          callbacks=[es],
          )

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
# accuracy_score :  0.10048
# f1_score :  0.09130561209653969

# 재현's SMOTE
# 
