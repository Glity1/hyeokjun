import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],[100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

# print(aaa, aaa.shape) # (13,2)
# [[   -10    100]
#  [     2    200]
#  [     3    -30]
#  [     4    400]
#  [     5    500]
#  [     6    600]
#  [     7 -70000]
#  [     8    800]
#  [     9    900]
#  [    10   1000]
#  [    11    210]
#  [    12    420]
#  [    50    350]]
# exit()

def outlier(data):
    n = data.shape[1]
    
    for i in range(n):
        nums = data[:, i]
        q_1, q_2, q_3 = np.percentile(nums, [25,50,75])   #quartile_1
        print('1사분위 : ', q_1)
        print('2사분위 : ', q_2)
        print('3사분위 : ', q_3)
        # exit()
        iqr = q_3 - q_1
        
        print('IQR : ', iqr)
        lower_bound = q_1 - (iqr*1.5)   # 1.5를 곱하는 이유는? 데이터마다 통계적으로 봤을 때 이상치의 위치를 봤을 때 표준정규분포를 따랐을 때 1.5로 생각했음.
        upper_bound = q_3 + (iqr*1.5)
        print(f'lower_bound :  {lower_bound} ')  # lower_bound :  -5.0
        print(f'upper_bound :  {upper_bound} ')  # upper_bound :  19.0
        # exit()
        outlier_loc = np.where((nums > upper_bound) | (nums < lower_bound))
        print('→ 이상치 인덱스:', outlier_loc)
        print('→ 이상치 값:', nums[outlier_loc])

outlier(aaa)

def outlier(nums):
    q1, q2, q3 = np.percentile(nums, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((nums > upper) | (nums < lower))
    return loc, iqr, upper, lower

for i in range(aaa.shape[1]):                                            # aaa.shape[1]의 의미는 2 즉 숫자, 총 열의 개수를 말한다 
    print(f"\n열 {i}에 대한 이상치 검사:")
    nums = aaa[:, i]
    outlier_loc, iqr, up, low = outlier(nums)
    print('이상치 위치:', outlier_loc)
    plt.figure()
    plt.boxplot(aaa[:, i])
    plt.axhline(up, color='red', linestyle='--', label='upper bound')
    plt.axhline(low, color='blue', linestyle='--', label='lower bound')
    plt.title(f'열 {i} boxplot')
    plt.legend()
    plt.show()
