import numpy as np
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input

import matplotlib.pyplot as plt

# def elu(x, alpha):
#     return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

elu = lambda x, alpha : (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))  # relu 에 비해서 연산량이 많음 

x= np.arange(-5,5,0.1) # -5~5range, 0.1 간격

print(x)
print(x.shape)  

y= elu(x, 0.5)   # elu : 양수 건드린다

plt.plot(x,y)
plt.title("elu Function")
plt.xlabel("x")
plt.ylabel("elu(x)")
plt.grid()
plt.show()

