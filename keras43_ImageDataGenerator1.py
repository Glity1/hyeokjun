import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train 160   test 120 
 
train_datagen = ImageDataGenerator(
    # ImageDataGenerator 이미지 데이터를 수치화 시켜준다. 
    rescale=1./255,                          # 0~255로 스케일링, 정규화                       
    horizontal_flip=True,                    # 수평 뒤집기 <- 데이터 증폭 또는 변환
    vertical_flip=True,                      # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1,                   # 평행이동 10% 
    height_shift_range=0.1,                   
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,                         # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환 (끌어당기거나 짜부시킴) 
    fill_mode='nearest'                      # 특정지역에 값이 소멸되는 곳에 대체값으로 근사값으로 넣어준다)
)                                 
 

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train'
path_test = './_data/image/brain/test'

xy_train = train_datagen.flow_from_directory(            # directory : 폴더
    path_train,
    target_size=(200, 200),                              # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=10,                                       # batch_size : default 32
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    seed=333,
)           

# Found 160 images belonging to 2 classes.  

xy_test = test_datagen.flow_from_directory(             # directory : 폴더
    path_test,
    target_size=(200, 200),                             # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale'
)      

#Found 120 images belonging to 2 classes.

# print(xy_train)
# #<keras.preprocessing.image.DirectoryIterator object at 0x000002A1A44162B0>

# print(xy_train[0])              # 총 160개의 데이터 중에 10개를 뽑아서 xy를 0번째에 집어넣었다.
# print(len(xy_train))            # 데이터의 길이 확인 : 16 /  0부터 15까지 들어가있음.
# print(xy_train[0][0].shape)     # (10, 200, 200, 1)
# print(xy_train[0][1].shape)     # (10,)
# print(xy_train[0][0].shape)    # (10, 200, 200, 1)
# print(xy_train[0][1].shape)     # (10,)
# print(xy_train[0][1])           # [0. 0. 1. 1. 1. 1. 0. 0. 0. 0.]

#x_train = xy_train[0][0]
#y_train = xy_train[0][1]
#통배치 작업을 하고싶을때 batch_size를 크게 넣는다 하지만 너무 크면 메모리가 힘들다.

# print(xy_train[0].shape)      # AttributeError: 'tuple' object has no attribute 'shape' 튜플형태는 .shape 사용불가
# print(xy_train[16])           # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[0][2])         # IndexError: tuple index out of range
# print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'> -> 0번째 배치의 x 데이터
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'> -> 0번째 배치의 y 데이터


