import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import GlobalAveragePooling2D


SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#1. 데이터
np_path = './_data/_save_npy/'

x = np.load(np_path + "keras46_07_x_train.npy") 
y = np.load(np_path + "keras46_07_y_train.npy") 

# plt.imshow(x[0])
# plt.show()

# exit()

# 2. train / val 나누기
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=777, stratify=y
)


vgg16 = VGG16(
    include_top=False,
    input_shape=(224, 224, 3), 
    # pooling='avg'
    )

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일 ,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=4,
    validation_data=(x_val, y_val),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss = model.evaluate(x_val, y_val)
results = model.predict(x_val)

# real = y_val[:20]

# print("예측값 : ", real)
print("loss : ", loss[0])
print("accuray : ", loss[1])

# 실습 # 
# 비교할것
# 1. 이전의 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False

# Flatten 과 GAP 비교

#데이터 하나당 4개 비교

# 이전결과


# 동결
# loss :  0.47073137760162354
# accuray :  0.7613292932510376

# 비동결
# loss :  0.6821609139442444
# accuray :  0.574018120765686

# Flatten 대신  GAP 사용
# loss :  0.6821752190589905
# accuray :  0.574018120765686
