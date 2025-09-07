from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

                                                # 세로, 가로, 색깔
model = Sequential()                            # height, width, channels
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))  #kernel = weight
model.add(Conv2D(filters=5, kernel_size=(2,2)))                       
model.add(Flatten())                            # 데이터 내용, 순서, 값이 바뀌지않는다.
model.add(Dense(units=10))                      # input = (batch_size, input_dim ) // Dense 2차원 입력 출력이나 4차원도 가능하다.
model.add(Dense(3)) 
                      
model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50        

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       

 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460       

 dense_1 (Dense)             (None, 3)                 33        

=================================================================
Total params: 748
Trainable params: 748
Non-trainable params: 0
_________________________________________________________________
"""





