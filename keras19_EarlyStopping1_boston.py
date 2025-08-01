# 18_1 cpoy
# 가독성 있게 배치할것

import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR) #(506,13) 데이터 DESCR : 묘사 (describe)
print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(506, 13)
print(y)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=714   # validation_split로도 바뀌지않는다면 바꾸자
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=20,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

hist = model.fit(x,y, epochs=10000, batch_size=16,      # loss, val_loss의 epochs의 수만큼 값을 반환해서 넣어준다 // 리스트 값이된다 2개니까
          verbose=1,                                  # return 이 포함된다.
          validation_split=0.2,
          callbacks=[es], 
          )
print('========================hist============================')
print(hist) # keras.callbacks.History object at 0x000002AE4E644220> 으로 나오는데 제대로 볼려면 
print('========================hist.history============================')
print(hist.history)  # 중괄호의 등장 : 키(loss, val_loss) : 벨류(숫자) 형태로 안에 넣어둔다 // loss, val loss 의 갯수는 epochs 값과 똑같음
                     # loss들의 역사 
                     # 그래프의 시각화가 가능하다 점들의 값이 있기 떄문에

print('========================hist.history에서 loss만 따로보고싶다============================')
print(hist.history['loss'])   # dictionary의 키값만 적어주면된다                     
       
print('========================hist.history에서 val_loss만 따로보고싶다============================')
print(hist.history['val_loss'])   # dictionary의 키값만 적어주면된다             

import matplotlib.pyplot as plt        # 맷플로립
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))         # 9X6사이즈로 만들어줘
plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
                                                                               # 리스트는 순서대로 가기때문에 x를따로 명시안해도된다. // y값만 넣으면 시간순으로 그림을 그림
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')               
plt.title('보스턴 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()


"""
#EarlyStopping 적용 후
    patience=20,     
) 

hist = model.fit(x,y, epochs=10000, batch_size=16,      
          verbose=1,                                  
          validation_split=0.2,
          callbacks=[es], 
          val_loss = 19.532060623168945
"""          