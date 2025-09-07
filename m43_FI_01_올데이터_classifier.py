#42_2 copy   // 각 데이터별로 feature_importances 뽑아냈음
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import accuracy_score
import time
import random
import numpy as np

seed = 742
random.seed(seed)
np.random.seed(seed)

#1. 데이터
data1 = load_breast_cancer()
data2 = load_wine()
data3 = load_digits()

datasets = [data1, data2, data3]
dataset_name = ['cancer', 'wine', 'digits']

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)
models = [model1, model2, model3, model4]

for i, data in enumerate(datasets) :
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("==============", dataset_name[i], "==============")

    #2. 모델구성
    for model in models:
        model.fit(x_train, y_train)
        
        print("===========", model.__class__.__name__, "===========")
        print('acc : ', model.score(x_test, y_test))
        print(model.feature_importances_)    
























