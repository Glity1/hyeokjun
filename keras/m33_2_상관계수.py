import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
df = pd.DataFrame({'X1' : [1,2,3,4,5],
                   'X2' : [5,4,3,2,1],
                   'X3' : [-1000,50,7,100,-1100],
                   'Y' : [2,3,4,5,6],
                   })

print(df)
#    X1  X2    X3  Y
# 0   1   5 -1000  2
# 1   2   4    50  3
# 2   3   3     7  4
# 3   4   2   100  5
# 4   5   1 -1100  6

correlations = df.corr()
print(correlations)
#          X1        X2        X3         Y
# X1  1.000000 -1.000000 -0.039156  1.000000
# X2 -1.000000  1.000000  0.039156 -1.000000
# X3 -0.039156  0.039156  1.000000 -0.039156
# Y   1.000000 -1.000000 -0.039156  1.000000
















