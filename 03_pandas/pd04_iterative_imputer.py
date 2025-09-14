import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.DataFrame(
    [[2, np.nan, 6, 8, 10],
     [2, 4, np.nan, 8, np.nan],
     [2, 4, 6, 8, 10],
     [np.nan, 4, np.nan, 8, np.nan],
     ]
).T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

imputer = IterativeImputer()        # default : BayesianRidge 회귀모델
data1 = imputer.fit_transform(data)
print(data1)
# [[ 2.          2.          2.          2.0000005 ]
#  [ 4.00000099  4.          4.          4.        ]
#  [ 6.          5.99999928  6.          5.9999996 ]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99999872 10.          9.99999874]]

imputer2 = IterativeImputer(estimator=xgb.XGBRegressor(max_depth=5,
                                                       learning_rate=0.1,
                                                       random_state=33
                                                       ),
                            max_iter=10,
                            random_state=33
                            )
data2 = imputer2.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          4.01184034]
#  [ 2.02664208  4.          4.          4.        ]
#  [ 6.          4.0039463   6.          4.01184034]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.98026466 10.          7.98815966]]

imputer3 = IterativeImputer(estimator=cat.CatBoostRegressor(depth=5,
                                                            learning_rate=0.5,
                                                            random_state=33
                                                            ),
                            max_iter=10,
                            random_state=33
                            )
data3 = imputer3.fit_transform(data)
print(data3)
# [[ 2.          2.          2.          4.        ]
#  [ 3.12851052  4.          4.          4.        ]
#  [ 6.          4.71184136  6.          4.52100891]
#  [ 8.          8.          8.          8.        ]
#  [10.          8.         10.          8.        ]]

imputer4 = IterativeImputer(estimator=lgb.LGBMRegressor(max_depth=5,
                                                        learning_rate=0.1,
                                                        random_state=33),
                            max_iter=10,
                            random_state=33
                            )
data4 = imputer4.fit_transform(data)
print(data4)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

imputer5 = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50,
                                                            max_depth=5,
                                                            random_state=33),
                            max_iter=10,
                            random_state=33
                            )
data5 = imputer5.fit_transform(data)
print(data5)
# [[ 2.    2.    2.    5.04]
#  [ 4.56  4.    4.    4.  ]
#  [ 6.    3.96  6.    5.04]
#  [ 8.    8.    8.    8.  ]
#  [10.    5.48 10.    6.24]]

imputer6 = IterativeImputer(estimator=GradientBoostingRegressor(max_depth=5,
                                                                n_estimators=20,
                                                                random_state=33),
                            max_iter=10,
                            random_state=33
                            )
data6 = imputer6.fit_transform(data)
print(data6)
# [[ 2.          2.          2.          4.24315331]
#  [ 2.8226332   4.          4.          4.        ]
#  [ 6.          3.57631667  6.          4.51869156]
#  [ 8.          8.          8.          8.        ]
#  [10.          5.66764873 10.          6.04224676]]

imputer7 = IterativeImputer(estimator=HistGradientBoostingRegressor(max_depth=5,
                                                                    max_iter=20,
                                                                    learning_rate=0.1,
                                                                    random_state=33),
                            max_iter=10,
                            random_state=33
                            )
data7 = imputer7.fit_transform(data)
print(data7)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

imputer8 = IterativeImputer(estimator=AdaBoostRegressor(n_estimators=10,
                                                        learning_rate=0.1,
                                                        random_state=33),
                            max_iter=10,
                            random_state=33
                            )
data8 = imputer8.fit_transform(data)
print(data8)
# [[ 2.  2.  2.  8.]
#  [ 2.  4.  4.  4.]
#  [ 6.  8.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

imputer9 = IterativeImputer(estimator=DecisionTreeRegressor(max_depth=5,
                                                         random_state=33),
                         max_iter=10,
                         random_state=33
                         )
data9 = imputer9.fit_transform(data)
print(data9)
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  4.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]