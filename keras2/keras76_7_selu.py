import numpy as np
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input

import matplotlib.pyplot as plt

alpha = 1.6732632423543772
lmbda = 1.0507009873554805

# def selu(x, alpha, lmbda):
#     return lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

selu = lambda x, alpha, lmbda : lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))  # relu 에 비해서 연산량이 많음 

x= np.arange(-5,5,0.1) # -5~5range, 0.1 간격

print(x)
print(x.shape)  

y= selu(x, 1, 2)    # selu : 양수 음수를 둘 다 건드린다

plt.plot(x,y)
plt.title("selu Function")
plt.xlabel("x")
plt.ylabel("selu(x)")
plt.grid()
plt.show()

