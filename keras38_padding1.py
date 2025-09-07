# [실습]
# 100,100,3 이미지를
# 10,10,11 으로 줄이자

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

# 모델구성
model = Sequential()
model.add(Conv2D(11, (2,2), input_shape=(100,100,3),
                 strides=10,
                 padding='valid',
                 ))
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 11)        143       

# =================================================================
# Total params: 143
# Trainable params: 143
# Non-trainable params: 0
# _________________________________________________________________

# 모델구성
model = Sequential()
model.add(Conv2D(5, (2,2), input_shape=(100,100,3),
                 strides=3,
                 ))
model.add(Conv2D(7,2,
                 strides=3,
                 ))
model.add(Conv2D(11,2,
                 strides=1,
                 ))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d_1 (Conv2D)           (None, 33, 33, 5)         65

#  conv2d_2 (Conv2D)           (None, 11, 11, 7)         147

#  conv2d_3 (Conv2D)           (None, 10, 10, 11)        319

# =================================================================
# Total params: 531
# Trainable params: 531
# Non-trainable params: 0
# _________________________________________________________________