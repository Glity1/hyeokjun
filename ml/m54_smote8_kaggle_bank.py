import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import random
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from tensorflow.python.keras.callbacks import EarlyStopping 



# 랜덤고정
seed = 123
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
# print(x.shape)  # (165034, 10)
y = train_csv['Exited']
# print(y.shape)

feature_names = list(x.columns)

print(x.shape, y.shape) # (165034, 10) (165034,)
print(np.unique(y, return_counts=True)) #(array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
print(pd.value_counts(y))
# 0    130113
# 1     34921
# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, shuffle=True,
    stratify=y
    )


smote = SMOTE(random_state=seed,
              k_neighbors=5,
            #  sampling_strategy='auto',  #default
            #  sampling_strategy=0.75,  # 최대값의 75% 지정
              # sampling_strategy={0:130113, 1:130113},  # (array([0, 1, 2]), array([50, 53, 33], dtype=int64))  # 샘플데이터를 제한하지말고 전부다 엄청 늘려버리면 과적합이긴하지만 성능은 좋음.
              sampling_strategy={0:150000, 1:150000},
              n_jobs=-1,  # 사용 가능한 모든 CPU 코어를 사용하라.   # imbleran 0.13버전 이상은  n_jobs는 안써도된다.
              )

x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))
#(array([0, 1], dtype=int64), array([130113, 130113], dtype=int64))
# exit()

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',  # ohe 안했음 
              optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

model.fit(x_train, y_train,
    epochs=400,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es],
    verbose=1,
    class_weight = class_weight_dict)

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
# accuracy_score :  0.788409801497855
# f1_score :  0.44084403968124897

# 재현's SMOTE
# accuracy_score :  0.788409801497855
# f1_score :  0.44084403968124897
