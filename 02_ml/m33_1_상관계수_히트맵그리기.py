import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets['target']

df = pd.DataFrame(x, columns=datasets.feature_names)      #numpy형태는 column이 따로없음 Dataframe형태로 만들어서 column을 생성
# print(df)
df['target'] = y
print(df)  # [150 rows x 5 columns]

print("================상관관계 히트맵 등장=========================")
print(df.corr())
# ================상관관계 히트맵 등장=========================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035          #  0.87이면 살짝 과적합 0.96은 과적합인데 둘중에 하나를 날려서 좋을수도있다
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547          #  x와 y 사이에는 상관관계 계수가 높을수록 좋음
# target                      0.782561         -0.426658           0.949035          0.956547  1.000000          # 대회에서는 -1같은 음의 계수는 날려버림

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()
















