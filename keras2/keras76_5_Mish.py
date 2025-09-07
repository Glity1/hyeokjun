import numpy as np
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input

import matplotlib.pyplot as plt

def mish(x):
    return x*np.tanh(np.log(1+np.exp(x)))

# mish = lambda x : x*np.tanh(np.log(1+np.exp(x)))  # relu 에 비해서 연산량이 많음 

x= np.arange(-5,5,0.1) # -5~5range, 0.1 간격

print(x)
print(x.shape)  

y= mish(x)

plt.plot(x,y)
plt.title("Mish Function")
plt.xlabel("x")
plt.ylabel("Mish(x)")
plt.grid()
plt.show()

