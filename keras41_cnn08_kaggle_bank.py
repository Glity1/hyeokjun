import numpy as np                                           
import pandas as pd                                                                               
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder #StandardScaler
import matplotlib.pyplot as plt        
import matplotlib.font_manager as fm
import matplotlib as mpl
from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자형 데이터 수치화
le = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

# 불필요한 col 제거
train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

# col 분리
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# 2차 col 분리
# binary_cols = ['Gender', 'Is Active Member','Has Cr Card']
# continuous_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
# print(y.value_counts())
#  0    130113
#  1     34921

# 데이터 스케일링
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# test_csv = scaler.transform(test_csv)

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=31,
)

print(x_train.shape, x_test.shape)  #(132027, 10) (33007, 10)
print(y_train.shape, y_test.shape)  #(132027,)  (33007,)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# exit()

x_train = x_train.reshape(132027,5,2,1)
x_test = x_test.reshape(33007,5,2,1)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(5, 2, 1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)
hist = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es],
    verbose=1,
    class_weight = class_weight_dict
)


# 4. 평가
results = model.evaluate(x_test, y_test, verbose=0)
print("loss : ", results[0])
print("accuracy : ", round(results[1], 5))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

roc_auc = roc_auc_score(y_test, y_predict)
print("ROC_AUC Score : ", roc_auc)