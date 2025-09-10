
# 예시 경로: C:\study25ju\_data\kaggle\dog_cat\train2

import numpy as np
import pandas as pd
import time

# 케라스 관련 모듈 (이미지 로딩 및 모델 구성용)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from sklearn.metrics import accuracy_score

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=10,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # brightness_range=[0.8, 1.2],
    # fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1./255)

# 🔸 데이터 경로 설정
path_train = './_data/kaggle/men_women/train'
path_test = './_data/kaggle/men_women/test'

# 🔹 데이터 로딩 시작 시간 측정
start = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,              # 훈련용 이미지 폴더 경로
    target_size=(224, 224),  # 모든 이미지를 224x224으로 리사이즈
    batch_size=64,          # 100장씩 불러오기
    class_mode='binary',     # 이진 분류용 라벨 (남/녀)
    color_mode='rgb',        # RGB 이미지
    shuffle=True             # 데이터를 섞어서 불러오기
)

# 🔸 테스트 이미지 제너레이터 정의
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
    # shuffle=True
)

# 🔸 불러온 데이터 확인
print(xy_train[0][0].shape)  
print(xy_train[0][1].shape)  
print(len(xy_train))         

# 🔹 로딩 시간 출력
end = time.time()
print("걸린 시간 :", round(end - start, 2), "초") 

# 🔹 전체 데이터를 저장할 리스트
all_x = []
all_y = []

# 🔸 각 배치마다 x, y 데이터를 리스트에 추가
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)

# 🔸 리스트를 넘파이 배열로 병합
x = np.concatenate(all_x, axis=0)  # 이미지 배열 합치기
y = np.concatenate(all_y, axis=0)  # 라벨 배열 합치기

# 🔸 전체 shape 출력
print("x.shape :", x.shape)  
print("y.shape :", y.shape)  

# 🔹 변환 시간 출력
end2 = time.time()
print("변환 시간 :", round(end2 - end, 2), "초")  # 예: 변환 시간 : 312.84 초

# 🔹 저장 시작 시간 측정
start2 = time.time()

# 🔸 넘파이 배열로 저장 (디스크에 저장하면 재사용 가능)
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + "keras46_07_x_train.npy", arr=x)  # 이미지 데이터 저장
np.save(np_path + "keras46_07_y_train.npy", arr=y)  # 라벨 데이터 저장

# 🔹 저장 종료 시간 출력
end3 = time.time()
print("npy 저장시간 :", round(end3 - start2, 2), "초")  # 예: npy 저장시간 : 407.52 초

