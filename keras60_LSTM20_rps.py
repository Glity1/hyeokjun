import numpy as np
import pandas as pd
import time

# 케라스 이미지 전처리 및 모델 구성 관련 모듈
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM,Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 정확도 평가 함수
from sklearn.metrics import accuracy_score

np_path = './_data/_save_npy/'

x = np.load(np_path + "keras46_05_x_train.npy")
y = np.load(np_path + "keras46_05_y_train.npy")

# 2. train / test 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print(x_train.shape)                        # (1638, 200, 200, 3)
print(x_train[0].shape)                     # (200, 200, 3) 

datagen = ImageDataGenerator(               # 클래스를 인스턴스화 하다.
    # rescale=1./255,                                                 
    horizontal_flip=True,                   # 좌우반전   
    # vertical_flip=True,                   # 상하반전
    width_shift_range=0.1,                 
    # height_shift_range=0.1,                   
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                       
    # fill_mode='nearest'                     # 이렇게 할꺼라고 준비상태
)      

augment_size = 1362                          
 
randidx = np.random.randint(x_train.shape[0], size=augment_size)     

print(randidx)                                
print(np.min(randidx), np.max(randidx))       # 1 2645

x_augmented = x_train[randidx].copy()         # 원본데이터 유지를 위해 copy()해서 새로운 공간으로 // 4만개의 데이터 copy, copy로 새로운 메모리 할당
                                              # 서로 영향 X
y_augmented = y_train[randidx].copy()         # x하고 같은 순서로 준비된다.

print(x_augmented)
print(x_augmented.shape)                      #(1362, 200, 200, 3)
print(y_augmented.shape)                      #(1362, 3)



x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3,
)

print(x_augmented.shape)                      #(1362, 200, 200, 3)

x_augmented_tmp, y_augmented_tmp = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()                                       # .next 없으면 iterator .next 있으면 batch_size의 형태로, .next()[0] 면 x_augmented 만 뽑고싶다면

print(x_augmented.shape)                           # (1362, 128, 128, 3)

print(x_train.shape)                               # (1638, 128, 128, 3)
x_train = x_train.reshape(-1, 128, 128, 3)
x_test = x_test.reshape(-1, 128, 128, 3)
print(x_train.shape, x_test.shape)                 # (1638, 128, 128, 3) (410, 128, 128, 3)

x_train = np.concatenate((x_train, x_augmented_tmp))   
y_train = np.concatenate((y_train, y_augmented_tmp))   
print(x_train.shape, y_train.shape)                # (3000, 128, 128, 3) (3000, 3)

x_train = x_train.reshape(x_train.shape[0], 128*128, 3)
x_test = x_test.reshape(x_test.shape[0], 128*128, 3)

# 3. 모델 구성
model = Sequential([
    LSTM(32, input_shape=(128*128,3)),
    BatchNormalization(), Activation('relu'),
    Dropout(0.2),
    
    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(3, activation='softmax')  # Binary Classification
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=256,
    validation_data=(x_test, y_test),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss = model.evaluate(x_test, y_test)

print("loss : ", loss[0])
print("accuray : ", loss[1])


# loss :  0.0043382602743804455
# accuray :  1.0