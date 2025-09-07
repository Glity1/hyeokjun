import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
# 지수분포의 평균(mean) 2.0

print(data)                                             # 
print(data.shape)                                       # (1000,)
print(np.min(data), np.max(data))                       # 0.00038263348751124216 18.462973956097333

log_data = np.log1p(data)                               # np.expm1p(data)
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()


























