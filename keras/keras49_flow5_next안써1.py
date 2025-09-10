from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

augment_size = 100                           # 증가시킬 사이즈

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)      # (60000,28,28)
print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()                # 신발 1개

# 신발 1개를 100개로 증폭

aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)   # (100, 28, 28, 1)

# bbb = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
# print(bbb.shape)   # (100, 28, 28, 1)

# exit()

datagen = ImageDataGenerator(               # 클래스를 인스턴스화 하다.
    rescale=1./255,                                                 
    horizontal_flip=True,                   # 좌우반전   
    # vertical_flip=True,                   # 상하반전
    width_shift_range=0.1,                 
    # height_shift_range=0.1,                   
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                       
    fill_mode='nearest'                     # 이렇게 할꺼라고 준비상태
)      

xy_data = datagen.flow(                       # 이미 수치화된 데이터라서 directory 안씀  #flow에서는 next를 범위 이상으로 써도 다시 돌아가는 로직이 있음. 즉 범위 안에서만 실행되게끔 해준다.
    # bbb,                                    # x,y 둘다 넣으면 tuple 형태 x 만 넣으면 numpy 형태
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),                 # y데이터 생성, augment_size에 해당하는 수로 0을 y값으로 채운다.
    batch_size=augment_size,                    
    shuffle=False,                                       
) #.next()                                    # 이 데이터의 첫번째를 뺀다. 그 다음꺼의 iterator가 실행됨

print(xy_data)                              # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001DA80701E50>
print(type(xy_data))                        # <class 'keras.preprocessing.image.NumpyArrayIterator'>

print(len(xy_data))                         # 1
print(xy_data[0][0].shape)                  # (100,28, 28, 1)  tuple' object has no attribute 'shape'
print(xy_data[0][1].shape)                  # (100,)  

# exit()

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][0][i], cmap='gray')
    
plt.show()

