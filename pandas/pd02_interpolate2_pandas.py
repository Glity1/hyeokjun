import numpy as np
import pandas as pd

data = pd.DataFrame(
    [[2, np.nan, 6, 8, 10],
     [2, 4, np.nan, 8, np.nan],
     [2, 4, 6, 8, 10],
     [np.nan, 4, np.nan, 8, np.nan],
     ]
)
print(data)
#      0    1    2  3     4
# 0  2.0  NaN  6.0  8  10.0
# 1  2.0  4.0  NaN  8   NaN
# 2  2.0  4.0  6.0  8  10.0
# 3  NaN  4.0  NaN  8   NaN

data = data.T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인
print(data.isna())
print(data.isna().sum())
print(data.isnull())
print(data.info())

# 결측치 삭제
print(data.dropna())    # 디폴트는 행, axis=0
#      0    1    2    3
# 3  8.0  8.0  8.0  8.0

print(data.dropna(axis=0)) # 행기준으로 결측치 있는 행은 다 삭제
#      0    1    2    3
# 3  8.0  8.0  8.0  8.0

print(data.dropna(axis=1)) # 열기준으로 결측치 있는 열은 다 삭제
#       2
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 특정값 - 평균
means = data.mean()   #각 열(Column)**에 대해 결측치(NaN)를 제외하고 평균을 계산합니다.
print(means)
# 0    6.500000
# 1    4.666667
# 2    6.000000
# 3    6.000000
# dtype: float64

data2 = data.fillna(means)
print(data2)
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0
exit()
# 특정값 - 중위
med = data.median()
print(med)
# 0    7.0
# 1    4.0
# 2    6.0
# 3    6.0
# dtype: float64

data3 = data.fillna(med)
print(data3)
#       0    1     2    3
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

# 특정값 - 임의지정
data4 = data.fillna(0)
print(data4)
#       0    1     2    3
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

data4_2 = data.fillna(99999)
print(data4_2)
#          0        1     2        3
# 0      2.0      2.0   2.0  99999.0
# 1  99999.0      4.0   4.0      4.0
# 2      6.0  99999.0   6.0  99999.0
# 3      8.0      8.0   8.0      8.0
# 4     10.0  99999.0  10.0  99999.0

# 시계열
data5 = data.ffill()    # 통상 마지막
print(data5)            # 첫번째 행은 채울 값이 없어서 Nan

data6 = data.bfill()    # 통상 첫번째
print(data6)            # 마지막 행은 채울 값이 없어서 Nan

################### 특정 컬럼만 ###################
means = data['x1'].mean()
print(means)            # 6.5

med = data['x4'].median()
print(med)              # 6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(med)

print(data)