import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import random
import matplotlib.pyplot as plt

# Seed 고정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, 
)

# 2. 모델 정의
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed, use_label_encoder=False, eval_metric='mlogloss')  # 경고 방지

models = [model1, model2, model3, model4]
model_names = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost']

# 3. 서브플롯 그리기
plt.figure(figsize=(12, 10))

for i, model in enumerate(models):
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"========== {model.__class__.__name__} ==========")
    print('acc : ', acc)
    print(model.feature_importances_)

    plt.subplot(2, 2, i+1)  # 2행 2열 중 i+1 번째 위치
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), [f'Feature {j}' for j in range(n_features)])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{model_names[i]} (acc={acc:.4f})")
    plt.ylim(-1, n_features)

plt.tight_layout()
plt.show()
