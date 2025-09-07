#2017년 구글 스위시란 이름으로 발표 표현된 SiLU(Sigmoid Linear Unit) 함수는
# 렐루(ReLU)와 시그모이드(sigmoid)의 장점을 결합한 함수입니다.
# SiLU는 입력값에 시그모이드 함수를 곱하여 음수 영역에서도 정보를 보존합니다.
# 이로 인해 렐루의 죽은 뉴런 문제를 완화하고, 시그모이드의 기울기 소실 문제를 개선합니다.
# SiLU는 자연스럽게 음수 영역에서도 활성화되어, 깊은 신경망에서도 안정적인 학습을 가능하게 합니다.
# SiLU는 렐루보다 더 부드러운 곡선을 가지며, 입력값이 0에 가까울 때도 기울기가 유지되어
# 학습이 원활하게 진행됩니다.  

import numpy as np
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input

import matplotlib.pyplot as plt

def silu(x):
    return x*(1/(1 + np.exp(-x)))

# silu = lambda x : x*(1/(1 + np.exp(-x)))  # SiLU는 입력값에 시그모이드 함수를 곱하여 음수 영역에서도 정보를 보존합니다.


x= np.arange(-5,5,0.1)


print(x)
print(x.shape)  #-5~5range, 0.1 간격

y= silu(x)

plt.plot(x,y)
plt.title("SiLU Function")
plt.xlabel("x")
plt.ylabel("SiLU(x)")
plt.grid()
plt.show()

