import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.decomposition import PCA

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

deep_list = [128,64,31,64,128]
# deep_list = [64,128,256,128,64]
# deep_list = [128,128,128,128,128]

def autoencoder(hidden_layer_list):
    model = Sequential([
        Dense(units=hidden_layer_list[0], input_shape=(28*28,)),
        Dense(units=hidden_layer_list[1], activation='relu'),
        Dense(units=hidden_layer_list[2], activation='relu'),
        Dense(units=hidden_layer_list[3], activation='relu'),
        Dense(units=hidden_layer_list[4], activation='relu'),
        Dense(784, activation='sigmoid')
    ])
    return model

# 0.95 이상  : 
# 0.99 이상  : 
# 0.999 이상 : 
# 1.0 이상   : 

model = autoencoder(hidden_layer_list=deep_list)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(
    x_train_noised, x_train,
    epochs=50, batch_size=128, validation_split=0.2, verbose=0
)
decoded_img= model.predict(x_test_noised, verbose=0)


""" 

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=8)
model_03 = autoencoder(hidden_layer_size=32)
model_04 = autoencoder(hidden_layer_size=64)
model_05 = autoencoder(hidden_layer_size=154)
model_06 = autoencoder(hidden_layer_size=331)
model_07 = autoencoder(hidden_layer_size=486)
model_08 = autoencoder(hidden_layer_size=713)

#3. 컴파일, 훈련
print('===================================================================')
model_01.compile(loss='binary_crossentropy', optimizer='adam')
model_01.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_02.compile(loss='binary_crossentropy', optimizer='adam')
model_02.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_03.compile(loss='binary_crossentropy', optimizer='adam')
model_03.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_04.compile(loss='binary_crossentropy', optimizer='adam')
model_04.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_05.compile(loss='binary_crossentropy', optimizer='adam')
model_05.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_06.compile(loss='binary_crossentropy', optimizer='adam')
model_06.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_07.compile(loss='binary_crossentropy', optimizer='adam')
model_07.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print('===================================================================')
model_08.compile(loss='binary_crossentropy', optimizer='adam')
model_08.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

#4. 평가, 예측
decoded_imgs_01 = model_01.predict(x_test_noised)
decoded_imgs_02 = model_02.predict(x_test_noised)
decoded_imgs_03 = model_03.predict(x_test_noised)
decoded_imgs_04 = model_04.predict(x_test_noised)
decoded_imgs_05 = model_05.predict(x_test_noised)
decoded_imgs_06 = model_06.predict(x_test_noised)
decoded_imgs_07 = model_07.predict(x_test_noised)
decoded_imgs_08 = model_08.predict(x_test_noised)
 """
 

import matplotlib.pyplot as plt
import random
fig, axes = plt.subplots(9,5, figsize=(15,15))
random_images = random.sample(range(decoded.shape[0], 5))

for row_num, row in enumerate(axes) :
    for col_num, ax in enumerate(row):
        ax.imshow(decoded[row_num][random_images[col_num]].shape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.show()