import numpy as np
import pandas as pd
from impyute.imputation.cs import mice

data = pd.DataFrame(
    [[2, np.nan, 6, 8, 10],
     [2, 4, np.nan, 8, np.nan],
     [2, 4, 6, 8, 10],
     [np.nan, 4, np.nan, 8, np.nan]]
).T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

data1 = mice(
    data.values,
    n=10,           # default : 5
    seed=777
    )
print(data1)
# [[ 2.  2.  2.  2.]
#  [ 4.  4.  4.  4.]
#  [ 6.  6.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10. 10. 10. 10.]]