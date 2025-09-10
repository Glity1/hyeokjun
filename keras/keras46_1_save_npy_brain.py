import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation

# train 160   test 120 
 
train_datagen = ImageDataGenerator(
    # ImageDataGenerator 이미지 데이터를 수치화 시켜준다. 
    rescale=1./255,                          # 0~255로 스케일링, 정규화                       
    horizontal_flip=True,                    # 수평 뒤집기 <- 데이터 증폭 또는 변환
    vertical_flip=True,                      # 수직 뒤집기 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1,                   # 평행이동 10% 
    # height_shift_range=0.1,                   
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,                       # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환 (끌어당기거나 짜부시킴) 
    fill_mode='nearest'                      # 특정지역에 값이 소멸되는 곳에 대체값으로 근사값으로 넣어준다)
)                                            # 데이터 증폭
 

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train'
path_test = './_data/image/brain/test'

start = time.time()

xy_train = train_datagen.flow_from_directory(            # directory : 폴더
    path_train,
    target_size=(200, 200),                              # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=160,                                       # batch_size : default 32
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    seed=333,
)           

# Found 160 images belonging to 2 classes.  

xy_test = test_datagen.flow_from_directory(             # directory : 폴더
    path_test,
    target_size=(200, 200),                             # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale'
)      

#Found 120 images belonging to 2 classes.

print(xy_train[0][0].shape)  #  : 첫 배치 이미지
print(xy_train[0][1].shape)  #  : 첫 배치 라벨
print(len(xy_train))

end = time.time()
print("걸린 시간 :", round(end - start, 2), "초")

all_x = []
all_y = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    
x = np.concatenate(all_x, axis=0)  # 이미지 배열 합치기
y = np.concatenate(all_y, axis=0)  # 라벨 배열 합치기

print("x.shape :", x.shape)  
print("y.shape :", y.shape)

end2 = time.time()
print("변환 시간 :", round(end2 - end, 2), "초")
 
start2 = time.time()

np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + "keras46_01_x_train.npy", arr=x)  # 이미지 데이터 저장
np.save(np_path + "keras46_01_y_train.npy", arr=y)

end3 = time.time()
print("npy 저장시간 :", round(end3 - start2, 2), "초")
   
end = time.time()
print("걸린시간 : ", end - start, '초')
