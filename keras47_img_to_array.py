from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img           # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array       # 가져온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np


path = 'c:/study25/_data/image/me/'

img = load_img(path + 'me.jpg', target_size=(224,224),)
print(img) #<PIL.Image.Image image mode=RGB size=100x100 at 0x2045B756E20>
print(type(img)) #<class 'PIL.Image.Image'>

plt.imshow(img)
plt.show()

arr = img_to_array(img)
img_array = img_to_array(img)
print(arr)
print(arr.shape)
print(type(arr))  # <class 'numpy.ndarray'>

# arr = arr.reshape(1,100,100,3)
# print(arr)
# print(arr.shape)

img = np.expand_dims(arr, axis=0)
print(img.shape) # (1, 200, 200, 3)

# me 폴더에 데이터를 npy로 저장한다.
np.save(path + 'keras47_me.npy', arr=img)
