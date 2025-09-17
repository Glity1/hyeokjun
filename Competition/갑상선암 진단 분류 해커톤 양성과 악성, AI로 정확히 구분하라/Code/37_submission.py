# https://dacon.io/competitions/official/236488/mysubmission

import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score

#1. 데이터
path = './_data/dacon/cancer/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
print(train_csv)                    # [87159 rows x 15 columns]             

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [46204 rows x 14 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [46204 rows x 1 columns] 

print(train_csv.shape)              #(87159,15)
print(test_csv.shape)               #(46204,14)
print(submission_csv.shape)         #(46204,1)
print(train_csv.columns)     #Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
                             #    'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
                             #    'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
                             #    'Cancer'],
                             #   dtype='object')

le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer'] 

print(train_csv.isna().sum())
print(test_csv.isna().sum())  #0

print(x) # [87159 rows x 14 columns]
print(y) # 87159

print(pd.DataFrame(y).value_counts())
print(pd.value_counts(y))

# 0         76700
# 1         10459

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True,
    )

scaler = StandardScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv.to_numpy().astype('float32'))

print(x_train.shape, x_test.shape)  # (69727, 14) (17432, 14)
print(y_train.shape, y_test.shape)  # (69727,) (17432,)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=14, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

def f1_score_keras(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics=['accuracy', f1_score_keras])

es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience=30,
    restore_best_weights=True,
)
hist = model.fit(x_train, y_train, epochs=300, batch_size=256,
          verbose = 2,
          validation_split= 0.2,
          callbacks=[es],
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)                                 # binary_crossentropy의 loss 와 accuracy를 볼 수 있다.
# [0.011887827888131142, 1.0]
# exit()

print(f"loss : {results[0]}")
print(f"accuracy : {results[1]:.5f}")
print(f"f1_score : {results[2]:.5f}")

y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict)                            # python 그냥 씀 / numpy 는 np.
# print(y_predict[:10])

accuracy_score= accuracy_score(y_test, y_predict)     # 변수 = 함수()
print("acc_score : ", accuracy_score)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
submission_csv['Cancer'] = y_submit
# print(submission_csv)

######################## csv파일 만들기 ######################
submission_csv.to_csv(path + 'submission_0529_1623.csv') # csv 만들기.

import matplotlib.pyplot as plt        # 맷플로립
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))  
plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
                                                                               # 리스트는 순서대로 가기때문에 x를따로 명시안해도된다. // y값만 넣으면 시간순으로 그림을 그림
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')              
plt.title('갑상선암 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()