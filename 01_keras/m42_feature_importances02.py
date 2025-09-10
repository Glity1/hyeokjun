# m42_01 copy
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_diabetes
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
# from sklearn. metrics import accuracy_score
# import time
# import random
# import numpy as np

# seed = 414
# random.seed(seed)
# np.random.seed(seed)

# #1. 데이터
# x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) # (442, 10) (442,)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=seed, #stratify=y
# )

# #2. 모델
# model1 = DecisionTreeRegressor(random_state=seed)
# model2 = RandomForestRegressor(random_state=seed)
# model3 = GradientBoostingRegressor(random_state=seed)
# model4 = XGBRegressor(random_state=seed)

# models = [model1, model2, model3, model4]
# for model in models:
#     model.fit(x_train, y_train)
    
#     print("===========", model.__class__.__name__, "===========")
#     print('r2 : ', model.score(x_test, y_test))
#     print(model.feature_importances_)    
    
# =========== DecisionTreeRegressor ===========
# r2 :  -0.24485184611248778
# [0.02743036 0.01152718 0.24971816 0.07130209 0.08311725 0.02915811
#  0.05384602 0.03000002 0.34096952 0.1029313 ]
# =========== RandomForestRegressor ===========
# r2 :  0.3686136245505226
# [0.05261665 0.01218362 0.29600997 0.10794567 0.04475532 0.06096236
#  0.05476272 0.02110838 0.27737498 0.07228033]
# =========== GradientBoostingRegressor ===========
# r2 :  0.3545175762238475
# [0.04071478 0.01537503 0.3018063  0.1098581  0.02463935 0.05420227
#  0.05236649 0.01239735 0.31211628 0.07652406]
# =========== XGBRegressor ===========
# r2 :  0.1600297839359559
# [0.02744878 0.0401694  0.18425536 0.08909228 0.06066815 0.0565148
#  0.07058903 0.07259717 0.342315   0.05635004]

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

seed_list = [74, 157, 888, 714, 310, 601, 512]

def get_models(seed):
    return [
        DecisionTreeRegressor(random_state=seed),
        RandomForestRegressor(random_state=seed),
        GradientBoostingRegressor(random_state=seed),
        XGBRegressor(random_state=seed, verbosity=0),
        LGBMRegressor(random_state=seed)
    ]

x, y = load_diabetes(return_X_y=True)

for seed in seed_list:
    print(f"\n===== 🔁 Seed = {seed} =====")
    np.random.seed(seed)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )

    models = get_models(seed)

    for model in models:
        model.fit(x_train, y_train)
        name = model.__class__.__name__
        r2 = model.score(x_test, y_test)
        importances = model.feature_importances_

        print(f"\n {name}")
        print(f" R2 Score: {r2:.5f}")
        print(f" Feature Importances: {np.round(importances, 4)}")
