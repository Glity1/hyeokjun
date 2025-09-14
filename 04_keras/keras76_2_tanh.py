import numpy as np
import matplotlib.pyplot as plt
import math
# x = np.arange(-5, 5, 0.1)

# y = np.tanh(x)

# plt.plot(x, y)
# plt.grid()
# plt.show()

tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# https://blog.naver.com/handuelly/221824080339  # 수식참고

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()