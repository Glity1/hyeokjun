#keras07_mlp2_1 copy  // model의 변경

import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.transpose(x)

print(x.shape) # (3, 10) -> (10, 3) 데이터구조를 받고나서 뒤에 숫자 2가 컬럼 input_dim=2가 된다.
print(y.shape) # (10,)

#2-1. 모델구성 (순차형)
model = Sequential()
model.add(Dense(10, input_dim=3)) 
model.add(Dense(9))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dense(1))

model.summary()
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                40

 dense_1 (Dense)             (None, 9)                 99

 dense_2 (Dense)             (None, 8)                 80

 dense_3 (Dense)             (None, 7)                 63

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
_________________________________________________________________

dropout 적용 후 (변화 없음) 실질적으로 모두 훈련시킨다. / 같은 모델이 여러번 바뀌는 효과를 가짐

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                40

 dropout (Dropout)           (None, 10)                0

 dense_1 (Dense)             (None, 9)                 99

 dense_2 (Dense)             (None, 8)                 80

 dropout_1 (Dropout)         (None, 8)                 0

 dense_3 (Dense)             (None, 7)                 63

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
_________________________________________________________________

"""
#2-2. 모델구성 (함수형)  # input layer 자체부터 구성을 시작한다
input1 = Input(shape=(3,))       # 대문자 클래스 를 인스턴스화 한다.
dense1 = Dense(10, name='ys1')(input1)       # (input) = input 에서 받아들인다 
dense2 = Dense(9, name='ys2')(dense1)        # name='원하는 이름 ' : summary에서 보이는 레이어의 이름을 지을 수 있음
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8)(drop1)         # 임의적으로 순서를 바꿔서 모델을 구성 가능하다.
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7)(drop2)
output1 = Dense(1)(dense4)
model2 = Model(inputs=input1, outputs=output1) #마지막에 input 부터 output까지 모델이라고 정의해준다.
model2.summary()


"""
_________________________________________________________________
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 3)]               0

 dense_5 (Dense)             (None, 10)                40

 dense_6 (Dense)             (None, 9)                 99

 dropout_2 (Dropout)         (None, 9)                 0

 dense_7 (Dense)             (None, 8)                 80

 dropout_3 (Dropout)         (None, 8)                 0

 dense_8 (Dense)             (None, 7)                 63

 dense_9 (Dense)             (None, 1)                 8

=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
_________________________________________________________________

"""






