from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# 원본은 n(데이터의 갯수),5,5,1 (1이니까 흑백)이미지
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # con layer 구성  (None, 4, 4, 10) 을 출력한다.
model.add(Conv2D(5, (2,2)))                       # (3, 3, 5)
model.add(Conv2D(3, (2,2)))                       # (2, 2, 3)

model.summary()
"""
 Model: "sequential"
 _________________________________________________________________
  Layer (type)                Output Shape              Param #
 =================================================================
  conv2d (Conv2D)             (None, 4, 4, 10)          50

 =================================================================
 Total params: 50
 Trainable params: 50
 Non-trainable params: 0
 _________________________________________________________________

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0
_________________________________________________________________

param : weight 와 bias 갯수

"""






