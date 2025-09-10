import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. 저장된 npy 데이터 불러오기
np_path = 'c:/study25/_data/_save_npy/'

x = np.load(np_path + "keras44_01_x_train.npy")
y = np.load(np_path + "keras44_01_y_train.npy")
print("x.shape:", x.shape, "| y.shape:", y.shape)

# 2. train / val 나누기
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(100, 100, 3)),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.2),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.3),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(), Activation('relu'),
    MaxPooling2D(), Dropout(0.4),

    Flatten(),
    Dense(32, activation='relu'), Dropout(0.5),
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
# 내 사진 경로
img_path = 'c:/study25/_data/image/me/'

# 내 사진 로드
x = np.load(img_path + "keras47_me.npy")

# 예측 수행
pred = model.predict(x)

if pred[0][0] > 0.5:
    print(f"내 사진은 개로 예측됨! ({pred[0][0]:.2f})")
else:
    print(f"내 사진은 고양이로 예측됨! ({1 - pred[0][0]:.2f})")
    