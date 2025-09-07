# https://dacon.io/competitions/official/236068/leaderboard

import numpy as np                                           
import pandas as pd                                          
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


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



#2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=8, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

path='./_save/keras28_mcp/07_dacon_당뇨병/'
# 체크포인트로 확인
model=load_model(path+'keras28_mcp_07_dacon_당뇨병.hdf5') 
# 세이브 모델 확인
model=load_model(path+'keras28_mcp_save_07_dacon_당뇨병.h5')

# path='./_save/keras28_mcp/07_dacon_당뇨병/'
# model.save_weights(path+'keras28_MCP_save_07_dacon_당뇨병.h5')

# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
#               metrics=['acc'])

# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#     monitor= 'val_loss',
#     mode= 'min',
#     patience=30,
#     restore_best_weights=True,
# )

# path='./_save/keras28_mcp/07_dacon_당뇨병/'
# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,    
#     filepath=path+'keras28_mcp_07_dacon_당뇨병.hdf5'
#     )

# hist = model.fit(x,y, epochs=500, batch_size=4,
#           verbose = 1,
#           validation_split= 0.2,
#           callbacks=[es, mcp],
#           )

# path='./_save/keras28_mcp/07_dacon_당뇨병/'
# model.save(path+'keras28_mcp_save_07_dacon_당뇨병.h5')

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

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
# submission_csv['Outcome'] = y_submit
# print(submission_csv)

######################## csv파일 만들기 ######################
#submission_csv.to_csv(path + 'submission_0527_1244.csv') # csv 만들기.

# import matplotlib.pyplot as plt        # 맷플로립
# import matplotlib.font_manager as fm
# import matplotlib as mpl

# font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
# font_name = fm.FontProperties(fname=font_path).get_name()
# mpl.rc('font', family=font_name)
# mpl.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(9,6))  
# plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
#                                                                                # 리스트는 순서대로 가기때문에 x를따로 명시안해도된다. // y값만 넣으면 시간순으로 그림을 그림
# plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')              
# plt.title('당뇨 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')  # 우측 상단에 label 표시
# plt.grid()                     # 격자표시
# plt.show()

"""
loss :  0.43404263257980347
accuracy :  0.77863
(131,) (131, 1)
(131, 1)
acc_score :  0.7786259541984732

"""
