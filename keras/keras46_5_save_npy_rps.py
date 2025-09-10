
# 예시 경로: C:\study25ju\_data\kaggle\dog_cat\train2

import numpy as np
import pandas as pd
import time

# 케라스 관련 모듈 (이미지 로딩 및 모델 구성용)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation

# 정확도 측정 함수
from sklearn.metrics import accuracy_score

# 🔸 이미지 전처리기 정의 (0~1 정규화)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 🔸 데이터 경로 설정
path_train = './_data/tensor_cert/rps/train'
path_test = './_data/tensor_cert/rps/test'

# 🔹 데이터 로딩 시작 시간 측정
start = time.time()

# 🔸 훈련 이미지 제너레이터 정의
xy_train = train_datagen.flow_from_directory(
    path_train,              # 훈련용 이미지 폴더 경로
    target_size=(200, 200),  # 모든 이미지를 200x200으로 리사이즈
    batch_size=100,          # 100장씩 불러오기
    class_mode='categorical',# 가위바위보 라벨 
    color_mode='rgb',        # RGB 이미지
    shuffle=True             # 데이터를 섞어서 불러오기
)

# 🔸 테스트 이미지 제너레이터 정의
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb',
    # shuffle=True
)

# 🔸 불러온 데이터 확인
print(xy_train[0][0].shape)  # (100, 200, 200, 3) : 첫 배치 이미지
print(xy_train[0][1].shape)  # (100,)             : 첫 배치 라벨
print(len(xy_train))         # 25000장 / 배치 100 → 총 250개의 배치

# 🔹 로딩 시간 출력
end = time.time()
print("걸린 시간 :", round(end - start, 2), "초") #걸린 시간 : 1.44 초

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
print("x.shape :", x.shape)  # 예: (25000, 200, 200, 3)
print("y.shape :", y.shape)  # 예: (25000,)

# 🔹 변환 시간 출력
end2 = time.time()
print("변환 시간 :", round(end2 - end, 2), "초")  # 예: 변환 시간 : 312.84 초

# 🔹 저장 시작 시간 측정
start2 = time.time()

# 🔸 넘파이 배열로 저장 (디스크에 저장하면 재사용 가능)
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + "keras46_05_x_train.npy", arr=x)  # 이미지 데이터 저장
np.save(np_path + "keras46_05_y_train.npy", arr=y)  # 라벨 데이터 저장

# 🔹 저장 종료 시간 출력
end3 = time.time()
print("npy 저장시간 :", round(end3 - start2, 2), "초")  # 예: npy 저장시간 : 407.52 초

#
# x.shape : (25000, 200, 200, 3)
# y.shape : (25000,)
# 변환 시간 : 312.84 초
# npy 저장시간 : 407.52 초

# 이 코드는 이미지 데이터를 제너레이터로 불러온 뒤,
# 전체 데이터를 메모리에 올리고, npy 파일로 저장하는 작업을 수행함
# 저장한 npy 파일은 추후에 빠르게 로드하여 학습에 사용할 수 있음
