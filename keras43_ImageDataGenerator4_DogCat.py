# 데이터 경로 예시: C:\study25ju\_data\kaggle\dog_cat\train2

import numpy as np
import pandas as pd
import time

# 케라스 이미지 데이터 로더 및 모델 구성 관련 모듈
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation

# 정확도 평가 함수
from sklearn.metrics import accuracy_score

# 🔹 이미지 전처리 (픽셀값을 0~1 사이로 정규화)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 🔹 이미지 경로 설정
path_train = './_data/kaggle/dog_cat/train2'
path_test = './_data/kaggle/dog_cat/test2'

# 🔹 데이터 로딩 시간 측정 시작
start = time.time()

# 🔹 훈련 데이터 불러오기
xy_train = train_datagen.flow_from_directory(
    path_train,              # 폴더 경로
    target_size=(200, 200),  # 모든 이미지를 200x200으로 크기 통일
    batch_size=100,          # 배치 크기 (100장씩 불러옴)
    class_mode='binary',     # 라벨 형식: 이진분류 (0 or 1)
    color_mode='rgb',        # 컬러 이미지
    shuffle=True             # 데이터 섞기
)

# 🔹 테스트 데이터 불러오기
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

# 🔹 확인용 출력
# 예시: (100, 200, 200, 3) => 100개의 200x200 RGB 이미지
print(xy_train[0][0].shape)  # 첫 번째 배치 이미지 데이터 형태
print(xy_train[0][1].shape)  # 첫 번째 배치 라벨 형태
print(len(xy_train))         # 전체 배치 수 (25000개 / 100 = 250)

# 🔹 데이터 로딩 시간 출력
end = time.time()
print("걸린 시간 :", round(end - start, 2), "초")

# 🔹 이미지와 라벨 데이터를 모두 모을 리스트
all_x = []
all_y = []

# 🔹 배치 단위로 전체 데이터를 리스트에 모음
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]   # 이미지와 라벨
    all_x.append(x_batch)
    all_y.append(y_batch)

# 🔹 리스트를 하나의 넘파이 배열로 병합 (차원 유지하면서 연결)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

# 🔹 병합 후 최종 shape 확인
print("x.shape :", x.shape)  # 예: (25000, 200, 200, 3)
print("y.shape :", y.shape)  # 예: (25000,)

# 🔹 병합 시간 출력
end2 = time.time()
print("변환 시간 :", round(end2 - end, 2), "초")  # 예: 변환 시간 : 295.8 초

# 중요 포인트:
# 이 작업은 `flow_from_directory` 로 배치로 나눠져 있는 데이터를 모두 메모리에 모으는 작업
# - 이 데이터를 `.npy` 파일 등으로 저장하면 다음부터는 빠르게 불러올 수 있음