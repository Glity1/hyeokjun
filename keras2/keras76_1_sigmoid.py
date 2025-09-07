import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
    # return 1 / (1 + np.exp(-x))  # exp 지수함수 e^-x  항상 0에서 1사이의 값이다

sigmoid = lambda x : 1 / (1 + np.exp(-x))   # 1회용으로 많이 씀

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()