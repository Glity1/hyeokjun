import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(-5, 5, 0.1)

alpha = 0.01  # Leaky ReLU의 기울기

def leaky_relu(x):
    # return np.maximum(alpha*x, x)
    return np.where(x > 0, x, alpha * x)  # 조건문

# leaky_relu = lambda x : np.where(x > 0, x, alpha * x)  # 조건문
# leaky_relu = lambda x : np.maximum(alpha*x, x)

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

