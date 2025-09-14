import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import os


SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#1. 데이터
np_path = './_data/_save_npy/'

x = np.load(np_path + "keras44_01_x_train.npy")
y = np.load(np_path + "keras44_01_y_train.npy")
# print("x.shape:", x.shape, "| y.shape:", y.shape)

# 2. train / test 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
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
# print(randidx)                                #[ 4731 18255  9328 ... 15349  1290 15380] 
# print(np.min(randidx), np.max(randidx))       # 0 19999

x_augmented = x_train[randidx].copy()         # 원본데이터 유지를 위해 copy()해서 새로운 공간으로 // 4만개의 데이터 copy, copy로 새로운 메모리 할당
                                              # 서로 영향 X
y_augmented = y_train[randidx].copy()         # x하고 같은 순서로 준비된다.

# print(x_augmented)
# print(x_augmented.shape)                      #(50000, 100, 100, 3)
# print(y_augmented.shape)                      #(50000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3,
)

# print(x_augmented.shape)                      #(50000, 100, 100, 3)

x_augmented_tmp, y_augmented_tmp = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()                                       # .next 없으면 iterator .next 있으면 batch_size의 형태로, .next()[0] 면 x_augmented 만 뽑고싶다면


# print(x_train.shape)                               # (20000, 100, 100, 3)
x_train = x_train.reshape(-1, 100, 100, 3)
x_test = x_test.reshape(-1, 100, 100, 3)
# print(x_train.shape, x_test.shape)                 # (20000, 100, 100, 3) (5000, 100, 100, 3)

# exit()

x_train = np.concatenate((x_train, x_augmented_tmp))   
y_train = np.concatenate((y_train,y_augmented_tmp))   
# print(x_train.shape, y_train.shape)                # (70000, 100, 100, 3) (70000,)

vgg16 = VGG16(
    include_top=False,
    input_shape=(100, 100, 3), 
    # pooling='flat'
    pooling='avg'
    )

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3. 컴파일 ,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(x_test, y_test),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss, acc = model.evaluate(x_test, y_test)
print(f"val_loss: {loss:.4f}, val_acc: {acc:.4f}")

###################################################################################


# ✅ 8. test2 이미지 예측
test_path = './_data/kaggle/dog_cat/test2/test/'
file_names = sorted(os.listdir(test_path))  # ex: 1.jpg, 2.jpg ...

x_pred = []
for fname in file_names:
    img = load_img(os.path.join(test_path, fname), target_size=(100, 100))
    img = img_to_array(img) / 255.0
    x_pred.append(img)

x_pred = np.array(x_pred)
print("x_pred.shape:", x_pred.shape)

y_pred_prob = model.predict(x_pred)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# 실습 # 
# 비교할것
# 1. 이전의 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False

# Flatten 과 GAP 비교

#데이터 하나당 4개 비교

# 이전결과
# loss :  0.6186747550964355
# acc :  0.7900999784469604
# accuracy_score :  0.7901

# 동결
# loss :  1.2354717254638672
# acc :  0.5709999799728394
# accuracy_score :  0.571

# 비동결

# Flatten 대신  GAP 사용

