import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# colunm 을 추가로 만들고 싶을 때
# 선형 -> 비선형으로 만들 때 효과적이다.

x = np.arange(12).reshape(4, 3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

pf = PolynomialFeatures(degree=3, include_bias=False)   #// Polynomial(다항식) # default = include_bias = True
x_pf = pf.fit_transform(x)
print(x_pf)
     
# 고차원은 잘 안쓴다.










