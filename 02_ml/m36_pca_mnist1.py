from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()             # 그 자리에 값을 받기는 해야하지만 값 자체는 필요없을 때 _ 로 표시
print(x_train.shape, x_test.shape)                        # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)             # 데이터 모으기
print(x.shape)                                            # (70000, 28, 28)

x = x.reshape(70000, 28*28)                               # (70000, 784) // 총784개의 데이터

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)

# # 1.0 이상인 첫 인덱스
# n1 = np.argmax(evr_cumsum >= 1.0)+1
# n1_sum = np.sum(evr_cumsum >=1.0)

# # 0.999 이상인 첫 인덱스
# n2 = np.argmax(evr_cumsum >= 0.999)+1 
# n2_sum = np.sum(evr_cumsum >=0.999)

# # 0.99 이상인 첫 인덱스
# n3 = np.argmax(evr_cumsum >= 0.99)+1
# n3_sum = np.sum(evr_cumsum >=0.99)

# # 0.95 이상인 첫 인덱스
# n4 = np.argmax(evr_cumsum >= 0.95)+1
# n4_sum = np.sum(evr_cumsum >=0.95)

# print(f"1.0 이상 : {n1}부터")
# print(f"0.999 이상 : {n2}부터")
# print(f"0.99 이상 : {n3}부터")
# print(f"0.95 이상 : {n4}부터")
# # 1.0 이상 : 712부터
# # 0.999 이상 : 485부터
# # 0.99 이상 : 330부터
# # 0.95 이상 : 153부터

# print(f"1.0 이상 : {n1_sum}개")
# print(f"0.999 이상 : {n2_sum}개")
# print(f"0.99 이상 : {n3_sum}개")
# print(f"0.95 이상 : {n4_sum}개")
# # 1.0 이상 : 72개
# # 0.999 이상 : 299개
# # 0.99 이상 : 454개
# # 0.95 이상 : 631개

value = [1.0, 0.999, 0.99, 0.95]

for i in value:
    nums = np.argmax(evr_cumsum >= i) + 1                     # 언제부터 원하는 값이 시작되는가?
    count = np.sum(evr_cumsum >= i)                         # 원하는 값의 개수는 총 몇인가?
    print(f"{i:.3f} 이상: {nums}번째부터, 총 {count}개")

# 1.000 이상: 713번째부터, 총 72개
# 0.999 이상: 486번째부터, 총 299개
# 0.990 이상: 331번째부터, 총 454개
# 0.950 이상: 154번째부터, 총 631개











