# 공분산과 평균을 이용해서 데이터를 타원형태의 군집으로 그리고, Mahalanobis 거리를 구해서 이상치를 찾는다.

import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],[100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)

# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]












