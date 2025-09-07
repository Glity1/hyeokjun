# 47 copy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img           # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array       # 가져온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np


path = 'c:/study25/_data/image/me/'

img = load_img(path + 'me.jpg', target_size=(100,100),)
print(img) #<PIL.Image.Image image mode=RGB size=100x100 at 0x2045B756E20>
print(type(img)) #<class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
img_array = img_to_array(img)
print(arr)
print(arr.shape)
print(type(arr))  # <class 'numpy.ndarray'>

# arr = arr.reshape(1,100,100,3)
# print(arr)
# print(arr.shape)

img = np.expand_dims(arr, axis=0)
print(img.shape) # (1, 100, 100, 3)

# me 폴더에 데이터를 npy로 저장한다.
# np.save(path + 'keras47_me.npy', arr=img)

#####################요기부터 증폭#############################

datagen = ImageDataGenerator(
    rescale=1./255,                                                 
    horizontal_flip=True,                   # 좌우반전   
    # vertical_flip=True,                   # 상하반전
    width_shift_range=0.1,                 
    # height_shift_range=0.1,                   
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                       
    fill_mode='nearest'                    
)      

it = datagen.flow(img,       
    batch_size=1,                                       
)           

print("====================================================================================")
print(it) #keras.preprocessing.image.NumpyArrayIterator object at 0x00000150691C4C40>
print("====================================================================================")

# 어떤 형태의 데이터인지 확인
# aaa = it.next()                            # python 2.0 문법
# print(aaa)
# print(aaa.shape) # (1, 100, 100, 3)

# # 많이 쓰는 방법
# bbb = next(it)
# print(bbb)
# print(bbb.shape) # (1, 100, 100, 3)

# 두개이상 쓰면 어떻게될까?

# print(it.next())     #iterator 갯수 이상만큼 next하면 오류가 뜬다 하지만 예외가있다. (numpyarrayiterator)순환해서 출력해줌
# print(it.next())
# print(it.next())

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5,5))
ax = ax.flatten() #2차원 -> 1차원 리스트로 변환

for i in range(10):
    # batch = it.next()  #IDG에서 랜덤으로 한번 작업 (변환)
    batch = next(it)
    print(batch.shape) #(1,100,100,3)
    batch = batch.reshape(100,100,3)  # 그림 그리기 위한 reshape 
    
    ax[i].imshow(batch)     # 찾아볼것
    ax[i].axis('off')       # 찾아볼것 
    
plt.show()