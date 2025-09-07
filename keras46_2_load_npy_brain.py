# 🔹 44_1 copy
# 저장된 이미지 데이터를 npy 파일로 불러와서 모델 학습 등에 사용할 준비를 함

import numpy as np
import pandas as pd
import time

# 케라스 이미지 전처리 및 모델 구성 관련 모듈
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np_path = 'c:/study25/_data/_save_npy/'

x = np.load(np_path + "keras46_01_x_train.npy")  # 이미지 데이터 (25000, 200, 200, 3)
y = np.load(np_path + "keras46_01_y_train.npy")  # 라벨 데이터 (25000,)


# 2. train / val 나누기
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(200, 200, 1)),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.2),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.3),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss, acc = model.evaluate(x_val, y_val)
print(f"val_loss: {loss:.4f}, val_acc: {acc:.4f}")
