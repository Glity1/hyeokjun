import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# 케라스 이미지 전처리 및 모델 구성 관련 모듈
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 정확도 평가 함수
from sklearn.metrics import accuracy_score

np_path = 'c:/study25/_data/_save_npy/'

x = np.load(np_path + "keras46_07_x_train.npy") 
y = np.load(np_path + "keras46_07_y_train.npy") 

# 2. train / val 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=777, stratify=y
)

print(x_train.shape)                        # (2647, 224, 224, 3)
print(x_train[0].shape)                     # (224, 224, 3) 

datagen = ImageDataGenerator(               # 클래스를 인스턴스화 하다.
    # rescale=1./255,                                                 
    horizontal_flip=True,                   # 좌우반전   
    vertical_flip=True,                   # 상하반전
    width_shift_range=0.1,                 
    height_shift_range=0.1,                   
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                       
    fill_mode='nearest'                     # 이렇게 할꺼라고 준비상태
)      
x_male = x_train[y_train == 0]
y_male = y_train[y_train == 0]

print(x_male.shape, y_male.shape)

augment_size = 494                         
randidx = np.random.randint(x_male.shape[0], size=augment_size)     # 6만개중에 4만개를 뽑아낸다 랜덤으로

print(randidx)                                # [2242 1196 1239 ... 1943 2271 1325] 
print(np.min(randidx), np.max(randidx))       # 1 2645

x_augmented = x_male[randidx].copy()         # 원본데이터 유지를 위해 copy()해서 새로운 공간으로 // 4만개의 데이터 copy, copy로 새로운 메모리 할당
                                              # 서로 영향 X
y_augmented = y_male[randidx].copy()         # x하고 같은 순서로 준비된다.

print(x_augmented)
print(x_augmented.shape)                      #(1353, 224, 224, 3)
print(y_augmented.shape)                      #(1353,)

# plt.imshow(x[0])
# plt.title(f"label: {y[0]}")
# plt.show()

# exit()
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3,
)

print(x_augmented.shape)                      #(1353, 224, 224, 3)

x_augmented_tmp, y_augmented_tmp = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/study25/_data/_save_img/08_men_women/'
).next()                                           # .next 없으면 iterator .next 있으면 batch_size의 형태로, .next()[0] 면 x_augmented 만 뽑고싶다면

exit()
print(x_train.shape)                               # (2647, 224, 224, 3)
x_train = x_train.reshape(-1, 224, 224, 3)
x_test = x_test.reshape(-1, 224, 224, 3)
print(x_train.shape, x_test.shape)                 # (2647, 224, 224, 3) (127104, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented_tmp))   
y_train = np.concatenate((y_train, y_augmented_tmp))   
print(x_train.shape, y_train.shape)                # (3141, 224, 224, 3) (4000,)

# 시각화
# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(x_train[randidx[0]])  # 타입변환 (혹시 float일 경우 대비)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Augmented")
# plt.imshow(x_augmented_tmp[0])
# plt.axis('off')

# plt.show()



# 3. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.2),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.2),
    
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.4),

    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

# real = y_val[:20]

# print("예측값 : ", real)
print("loss : ", loss[0])
print("accuray : ", loss[1])

# loss :  0.4804737865924835
# accuray :  0.773413896560669
