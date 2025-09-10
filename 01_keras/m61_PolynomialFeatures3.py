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

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
# degree 차원수 결정
x_pf = pf.fit_transform(x)
print(x_pf)
#[[  0.   1.   2.   0.   0.   2.]    # 상관계수, feature importance, pca 로 확인한다
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]