import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# 랜덤고정
seed = 123
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']    

feature_names = list(x.columns)

le = LabelEncoder() 
y = le.fit_transform(y)

print(x.shape, y.shape) #(61878, 93) (61878,)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))
print(pd.value_counts(y))
# 1    16122
# 5    14135
# 7     8464
# 2     8004
# 8     4955
# 6     2839
# 4     2739
# 3     2691
# 0     1929

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=seed, train_size=0.8,  stratify=y
    )

ros = RandomOverSampler(random_state=seed,
              sampling_strategy='auto',  #default
              # sampling_strategy={0:200000, 1:200000, 2:200000, 3:200000, 4:200000, 5:200000, 6:200000, 7:200000, 8:200000},
              # n_jobs=-1,  # 사용 가능한 모든 CPU 코어를 사용하라.   # imbleran 0.13버전 이상은  n_jobs는 안써도된다.
              )

x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([12898, 12898, 12898, 12898, 12898, 12898, 12898, 12898, 12898],

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(128, input_shape=(93,), activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.05))   

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.1))   

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.1))   

model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',  # ohe 안했음 
              optimizer='adam', metrics=['acc'])

es = EarlyStopping(          
    monitor='val_loss',
    mode = 'min',            
    patience=20, # patience를 50으로 더 증가
    restore_best_weights=True,    
) 

model.fit(x_train, y_train, epochs=100, batch_size=256,
          verbose =2,
          validation_split=0.2,
          callbacks=[es],
          )

# print(np.unique(y_train, return_counts=True))

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
# accuracy_score :  0.7969457013574661
# f1_score :  0.7659881827493988

# 재현's SMOTE
# accuracy_score :  0.7850678733031674
# f1_score :  0.7531828195407798

# accuracy_score :  0.7772301228183581
# f1_score :  0.7495280946489875
