import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer



#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)                    # (569,30)
                                         # Missing Attribute Values: None (결측치없음)
print(datasets.feature_names)               

print(type(datasets))                    # <class 'sklearn.utils.Bunch'>

x = datasets.data
y = datasets.target

print(x.shape, y.shape)                  # (569, 30) (569,)
print(type(x))                           # <class 'numpy.ndarray'>

print(x)
print(y)                                 # 0,1 맞추기 분류

# 0과 1의 개수가 몇개인지 찾아보기.

print(np.unique(y,return_counts=True))  #1. numpy 로 찾았을 때
# (array([0, 1]), array([212, 357], dtype=int64))

print(pd.value_counts(y))               #2. pandas 로 찾았을 때
# 1    357
# 0    212
# dtype: int64

#print(y.value_counts())                               #3. numpy는 value_counts가 없음 y를 pandas로 바꿔야함

print(pd.DataFrame(y).value_counts())                  #4. numpy는 value_counts가 없음 y를 pandas로 바꿔야함 #데이터프레임형태 2차원 테이블 형태의 자료 구조
print(pd.Series(y).value_counts())                     # pandas의 1차원 데이터 구조 벡터형태

# 1    357                                             # 불균형 데이터는 아님.
# 0    212
# dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=384,
    shuffle=True,
)

print(x_train.shape, x_test.shape)  #(398, 30) (171, 30)
print(y_train.shape, y_test.shape)  #(398,) (171,)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))                                # 데이터손실 때문에 relu를 쓴다
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,  activation='sigmoid'))                             # 활성화 함수를 한정하고 제한해야한다 // 0.5를 기준으로 오른쪽은 1 왼쪽으로 0
                                                                       # 이진분류는 마지막 output에 activation = 'sigmoid' 쓴다. 마지막 load 는 항상 1이다.
# path='./_save/keras28_mcp/06_cancer/'
# model.save_weights(path+'keras28_MCP_save_06_cancer.h5')

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])             #binary: 이진법  crossentropy : 엔트로피(특정 분포 내에서의 정보량의 기댓값을 의미한다는 것)를 서로 다른 확률 분포에서 계산한 값을 의미합니다.
                                                                        # loss 값이 크면 나쁘고 작으면 좋다.
                                                                        # metrics=['acc']/ accuracy : 정확성을 훈련과정에서 보여줄려고 
start_time = time.time()                                                                       
from tensorflow.python.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                    # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=10,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,         # 가장 최소 지점으로 저장한다
) 

path='./_save/keras28_mcp/06_cancer/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras33_mcp_06_cancer.hdf5'
    )

hist = model.fit(x,y, epochs=50, batch_size=16,   
          verbose =1,
          validation_split=0.2,
          callbacks=[es, mcp],                      # 나중에 [es]안에 추가될수있음.
          )
end_time = time.time()

path='./_save/keras28_mcp/06_cancer/'
model.save(path+'keras33_mcp_save_06_cancer.h5')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)       
print(results)                                 # binary_crossentropy의 loss 와 accuracy를 볼 수 있다.
# [0.011887827888131142, 1.0]
# exit()

print("loss : ", results[0])                   # loss : 0.022432664409279823
print("accuracy : ",round(results[1], 5))      # acc  : 0.987246

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)                            # python 그냥 씀 / numpy 는 np.
print(y_predict[:10])

# exit()

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
accuracy_score  = accuracy_score(y_test, y_predict)     # 변수 = 함수()
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2))

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
plt.title('캔서 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()

"""

loss :  0.08258756995201111
acc_score :  0.9649122807017544
걸린시간 :  1.64

dropout 적용 후
acc_score :  0.8771929824561403
"""                                                                              
                                                                        
                                                                        
                                                                        
                                                                        
                                                                        