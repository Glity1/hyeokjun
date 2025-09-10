
import numpy as np
import sklearn as sk
print(sk.__version__) # 1.1.3
import tensorflow as tf
print(tf.__version__) # 2.9.3

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names) #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target #통상적으로 target 은 y값

print(x)
print(y)
print(x.shape, y.shape) # x 형태 (20640, 8) y 형태 (20640,) 행은 항상 같다
                        # 소수점으로 되있으면 회귀모델
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813 # validation_split로도 바뀌지않는다면 바꾸자
)

print(x_train.shape, y_train.shape)  # (16512, 8) (16512,)
print(x_test.shape, y_test.shape)    # (4128, 8) (4128,)

# exit()

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 


x_train = x_train.reshape(16512,2,2,2)
x_test = x_test.reshape(4128,2,2,2)

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
model.compile(loss= 'mse', optimizer= 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=30,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

# path='./_save/keras28_mcp/02_california/'
# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,    
#     filepath=path+'keras35_mcp_02_california.hdf5'
#     )

hist = model.fit(x_train,y_train, epochs=200, batch_size=64,
          verbose =1,
          validation_split=0.2,
          callbacks=[es],
          ) 

# path='./_save/keras28_mcp/02_california/'
# model.save(path+'keras35_mcp_save_02_california.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x_test]의 예측값 : ", results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)

# import matplotlib.pyplot as plt        # 맷플로립
# import matplotlib.font_manager as fm
# import matplotlib as mpl

# font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
# font_name = fm.FontProperties(fname=font_path).get_name()
# mpl.rc('font', family=font_name)
# mpl.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(9,6))         # 9X6사이즈로 만들어줘
# plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
#                                                                                # 리스트는 순서대로 가기때문에 x를따로 명시안해도된다. // y값만 넣으면 시간순으로 그림을 그림
# plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')               
# plt.title('켈리포니아 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')  # 우측 상단에 label 표시
# plt.grid()                     # 격자표시
# plt.show()

"""
loss :  2.5319135189056396
[x_test]의 예측값 :  [[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
r2 스코어 :  -0.885554581579648
          
"""