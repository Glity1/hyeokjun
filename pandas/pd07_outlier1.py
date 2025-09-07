import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,13,14,15,50])

def outlier(data):
    q_1, q_2, q_3 = np.percentile(data, [25,50,75])   #quartile_1
    print('1사분위 : ', q_1)
    print('2사분위 : ', q_2)
    print('3사분위 : ', q_3)
    iqr = q_3 - q_1
    
    print('IQR : ', iqr)
    lower_bound = q_1 - (iqr*1.5)   # 1.5를 곱하는 이유는? 데이터마다 통계적으로 봤을 때 이상치의 위치를 봤을 때 표준정규분포를 따랐을 때 1.5로 생각했음.
    upper_bound = q_3 + (iqr*1.5)
    print(f'lower_bound :  {lower_bound} ')  # lower_bound :  -5.0
    print(f'upper_bound :  {upper_bound} ')  # upper_bound :  19.0
    # exit()
    return np.where((data > upper_bound) | (data < lower_bound)), \
    iqr, lower_bound, upper_bound

outlier_loc, iqr, up, low, = outlier(aaa)
print('이상치의 위치 : ', outlier_loc)

# 1사분위 :  4.0
# 2사분위 :  7.0
# 3사분위 :  10.0
# IQR :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(up, color= 'pink', label='upper bound')
plt.axhline(low, color= 'pink', label='lower bound')
plt.legend()
plt.show()

# 털 부분 이상치를 제외한 값 중에 최댓값, 최솟값
# boxplot 은 행렬형태의 데이터도 받는다. (여러모로 사용하기 좋음.)
