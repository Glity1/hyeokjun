from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

#. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=1,
                 padding='same',
                #  padding='valid',
                 ))
model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=1,
                 padding='valid'           # padding = 'valid' 가 default
                 ))
model.add(Conv2D(18, 4))                    # (8, 4) =  (filters=8, kernel_size(4,4))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 10)        50

#  conv2d_1 (Conv2D)           (None, 8, 8, 9)           819

#  conv2d_2 (Conv2D)           (None, 5, 5, 18)          2610

# =================================================================
# Total params: 3,479
# Trainable params: 3,479
# Non-trainable params: 0
# _________________________________________________________________

