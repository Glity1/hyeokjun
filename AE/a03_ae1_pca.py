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

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

print('1 이상 : ', np.argmax(evr_cumsum>=1.0)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)

def autoencoder(hidden_layer_size):
    model = Sequential([
        Dense(units=hidden_layer_size, input_shape=(28*28,)),
        Dense(784, activation='sigmoid')
    ])
    return model

# 0.95 이상  : 
# 0.99 이상  : 
# 0.999 이상 : 
# 1.0 이상   : 

hidden_size = 64
hidden_size = 64
hidden_size = 64
hidden_size = 64
model = autoencoder(hidden_layer_size=hidden_size)

#3. 컴파일, 훈련
# autoencoder.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam')   # 0과 1이기 때문에 가능
model.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, )

#4. 평가, 예측
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