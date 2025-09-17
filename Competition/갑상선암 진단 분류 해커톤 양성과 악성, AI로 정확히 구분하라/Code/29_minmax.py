# https://dacon.io/competitions/official/236488/mysubmission

import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
#1. 데이터
path = './_data/dacon/cancer/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) # 0번째 컬럼을 인덱스로

le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=93, shuffle=True,
    )


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv.to_numpy().astype('float32'))

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=14, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])

es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience=30,
    restore_best_weights=True,
)
hist = model.fit(x_train, y_train, epochs=500, batch_size=128,
          verbose = 1,
          validation_split= 0.2,
          callbacks=[es],
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)               


f1 = f1_score(y_test, y_predict)
print("f1_score : ", f1)

accuracy_score= accuracy_score(y_test, y_predict)    
print("acc_score : ", accuracy_score)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

submission_csv['Cancer'] = y_submit
submission_csv.to_csv(path + 'submission_0529_1531.csv') # csv 만들기.

import matplotlib.pyplot as plt        # 맷플로립
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))  
plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')              
plt.title('갑상선암 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()