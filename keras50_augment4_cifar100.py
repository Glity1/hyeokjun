import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# x_train = x_train/255.
# x_test = x_test/255.

print(x_train.shape)                        
print(x_train[0].shape)                      

datagen = ImageDataGenerator(               # 클래스를 인스턴스화 하다.
    rescale=1./255,                                                 
    horizontal_flip=True,                   # 좌우반전   
    # vertical_flip=True,                   # 상하반전
    width_shift_range=0.1,                 
    # height_shift_range=0.1,                   
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                       
    # fill_mode='nearest'                     # 이렇게 할꺼라고 준비상태
)      

augment_size = 50000                          # 6만개 -> 10만개로 만들거야. 4만개를 추가로 늘리니까
 
randidx = np.random.randint(x_train.shape[0], size=augment_size)     # 6만개중에 4만개를 뽑아낸다 랜덤으로
          # np.random.randint(60000, 40000) // 윗줄과 같다
print(randidx)                                # [50246 34574 51686 ... 35126 43098  8230] 
print(np.min(randidx), np.max(randidx))       # 1 59999

x_augmented = x_train[randidx].copy()         # 원본데이터 유지를 위해 copy()해서 새로운 공간으로 // 4만개의 데이터 copy, copy로 새로운 메모리 할당
                                              # 서로 영향 X
y_augmented = y_train[randidx].copy()         # x하고 같은 순서로 준비된다.

print(x_augmented)
print(x_augmented.shape)                      #(50000, 28, 28, 3)
print(y_augmented.shape)                      #(50000, 1)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3,
)

print(x_augmented.shape)                      #(50000, 32, 32, 3)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]                                        # .next 없으면 iterator .next 있으면 batch_size의 형태로, .next()[0] 면 x_augmented 만 뽑고싶다면

print(x_augmented.shape)                           # (50000, 32, 32, 3)



print(x_train.shape)                               # (50000, 32, 32, 3)
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(x_train.shape, x_test.shape)                 # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))   
print(x_train.shape, y_train.shape)                # (100000, 32, 32, 3) (100000, 1)

y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape)

#2. 모델구성

model = Sequential()
model.add(Conv2D(256, (2,2), strides=2, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(256, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (2,2), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(128, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(units=100, activation='softmax'))

#3. 컴파일 ,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode = 'min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=5000, batch_size=64,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc

y_pred = model.predict(x_test)
y_test = y_test.values
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)  

acc= accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

# loss :  3.108353853225708
# acc :  0.2583000063896179
# accuracy_score :  0.2583