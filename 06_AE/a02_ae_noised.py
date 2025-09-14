import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

#2. 모델
input_img = Input(shape=(28*28,))

#### 인코더 ####
encoder = Dense(1, activation='relu')(input_img)
# encoder = Dense(32, activation='relu')(input_img)
# encoder = Dense(64, activation='relu')(input_img)
# encoder = Dense(128, activation='relu')(input_img)
# encoder = Dense(256, activation='relu')(input_img)
# encoder = Dense(512, activation='relu')(input_img)
# encoder = Dense(784, activation='relu')(input_img)
# encoder = Dense(1024, activation='relu')(input_img)

#### 디코더 ####
decoder = Dense(28*28, activation='sigmoid')(encoder)

autoencoder = Model(input_img, decoder)

#3. 컴파일 훈련
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

#4. 평가 예측
decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(15, 4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()