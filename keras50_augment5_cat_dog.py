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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 저장된 npy 데이터 불러오기
np_path = 'c:/study25/_data/_save_npy/'

x = np.load(np_path + "keras44_01_x_train.npy")
y = np.load(np_path + "keras44_01_y_train.npy")
print("x.shape:", x.shape, "| y.shape:", y.shape)

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
).next()                                       # .next 없으면 iterator .next 있으면 batch_size의 형태로, .next()[0] 면 x_augmented 만 뽑고싶다면


print(x_train.shape)                               # (50000, 32, 32, 3)
x_train = x_train.reshape(-1, 100, 100, 3)
x_test = x_test.reshape(-1, 100, 100, 3)
print(x_train.shape, x_test.shape)                 # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))   
print(x_train.shape, y_train.shape)                # (100000, 32, 32, 3) (100000, 1)

y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape)


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


