import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv3D, MaxPooling2D, UpSampling2D

#1. 데이터
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
                                 # (평균, 표준편차 0.1인 정규분포형태의 랜덤값, size)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)        # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.4611409472854433 -0.5246125806788717

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.0 0.0

#2. 모델
input_img = Input(shape=(28*28,))
##### 인코더 #####
x = Conv3D(64, (2,2), padding='same')(input_img)
x = MaxPooling2D()(x)
x = Conv3D(32, (2,2), padding='same')(x)
x = MaxPooling2D()(x)

##### 디코더 #####
x = Conv3D(32, (2,2), padding='same')(x)
x = UpSampling2D()(x)
x = Conv3D(16, (2,2), padding='same')(x)
x = UpSampling2D()(x)
out = Conv3D(1, (2,2), padding='same')(x)

model = Model(input_img, out)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train_noised, x_train, epochs=30, batch_size=256, validation_split=0.2)

decoded_img = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(15, 4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()