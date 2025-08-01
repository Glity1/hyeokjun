import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes                               #당뇨병 diabetes
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=814 # validation_split로도 바뀌지않는다면 바꾸자
)

#2. 모델구성
model = Sequential()
model.add(Dense (64, input_dim=10))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')

from tensorflow.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=100,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

hist = model.fit(x_train, y_train, epochs=200, batch_size=1,
          verbose =1,
          validation_split=0.25,
          callbacks=[es]
          )

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss : ', loss)
print("[x_test]의 예측값 : ", results)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)

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
plt.title('당뇨병 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()