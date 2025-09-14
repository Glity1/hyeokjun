# 34_2 copy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

### feature 를 압축하는 개념 ### // 돼지 -> 소세지  // 몇 개로 압축할꺼냐?? 를 선정 잘해야함.

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
    stratify=y
)

######### pca하기전에 scaler하는게 좋다 ###############
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier(random_state=222)
    
for i in range(x.shape[1]) : 
    pca = PCA(n_components=i+1)                     
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    model.fit(x_train_pca, y_train)
    score = model.score(x_test_pca, y_test)
    print(f"[PCA n_components={i+1}] Test Score: {score}")
    
evr = pca.explained_variance_ratio_              # 설명가능한 변화율
print('evr : ', evr)                             # evr :  [0.73244098 0.22584625 0.0363096  0.00540317]   # 1~4개까지 썼을 때 pca자체에 미치는 영향률
print('evr_sum : ', sum(evr))                    # evr_sum :  1.0

## n_components를 몇개까지 써야 효율적인지 알 수 있다.
############# 누적 합 #################

evr_cumsum = np.cumsum(evr)
print('evr_cumsum : ', evr_cumsum)               # evr_cumsum :  [0.73244098 0.95828723 0.99459683 1.        ]
                                                 # n_components 1,2,3,4일때의 영향률

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

















































