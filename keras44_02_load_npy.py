import numpy as np
import pandas as pd
import time
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from sklearn.metrics import accuracy_score

np_path = 'c:/study25/_data/_save_npy/'

start = time.time()

x_train = np.load(np_path + "keras44_01_x_train.npy")  
y_train = np.load(np_path + "keras44_01_y_train.npy")  

end = time.time()

print(x_train)              
print(y_train[:20])         

print(x_train.shape, y_train.shape)  

print("불러오기 걸린시간 :", round(end - start, 2), "초")
