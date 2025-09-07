import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import accuracy_score
import time
import random
import numpy as np

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

#2. 모델
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    
    print("===========", model.__class__.__name__, "===========")
    print('acc : ', model.score(x_test, y_test))
    print(model.feature_importances_)    
    
#=========== DecisionTreeClassifier ===========
# acc :  0.9
# [0.02916667 0.         0.91887218 0.05196115] 
# =========== RandomForestClassifier ===========
# acc :  0.9666666666666667
# [0.10870669 0.01823337 0.51102309 0.36203684] 
# =========== GradientBoostingClassifier ===========
# acc :  0.9666666666666667
# [0.00709144 0.0123877  0.76481073 0.21571013]     
# =========== XGBClassifier ===========
# acc :  0.9333333333333333
# [0.01646412 0.01143489 0.8465672  0.12553383]
























