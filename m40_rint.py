import numpy as np

x = np.array([[15.2], [38.7], [60.3], [82.8], [99.9]])

x = np.rint(x).astype(int)

print(x) 
#[[ 15.]   # 소수점 형태 :  x = np.rint(x)
#  [ 39.] 
#  [ 60.] 
#  [ 83.] 
#  [100.]]

# [[ 15]   # 정수형 형태 : x = np.rint(x).astype(int)
#  [ 39]
#  [ 60]
#  [ 83]
#  [100]]