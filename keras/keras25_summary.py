import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary() # 어떻게 연산됐는지 보여줌
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 4)                 12

#  dense_3 (Dense)             (None, 1)                 5

# =================================================================   
# Total params: 31          
# Trainable params: 31       이건 왜 필요할까? 
# Non-trainable params: 0    훈련을 시키지않아도 되는 모델// 전이 학습 // 가중치를 그대로 쓴다고 생각하면 훈련필요x
# _________________________________________________________________







