import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1,2,3,1],                        # 이 부분이 x 에 해당하는부분
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]]
                )

print(data.shape)  #(5,4)

#1) 평균
means = np.mean(data, axis=0)  #(5,4)
print("평균 : ", means) # 평균 : [7. 8. 9. 47.]

#2) 모집단 분산 (n으로 나눈다) = 분산 : mse와 비슷함 ()
population_variances = np.var(data, axis=0)
print("모집단 분산 : ", population_variances)       # 모집단 분산 :  [  18.   18.   18. 3038.]

#3) 표본 분산 (n-1로 나눈다)
variances = np.var(data, axis=0, ddof=1)           # ddof : n-1빵 하겠다.
print("표본분산 : ", variances)                     # 표본분산 :  [  22.5   22.5   22.5 3797.5]

#4) 표본 표준편차
std1 = np.std(data, axis=0, ddof=1)                # 표본 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]
print("표본 표준편차 : ", std1)

#5) 모집단 표준편차
std2 = np.std(data, axis=0)                        # 통상적으로 np.std 모집단 표준편차
print("모집단 표준편차 : ", std2)                   # 모집단 표준편차 :  [ 4.24264069  4.24264069  4.24264069 55.11805512]

#6) Standardscaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('standardScaler : \n', scaled_data)                    # \n 줄바꿈 , 통상 표편은 모집단 표편이다

# 계산식 x(#1의 데이터)-평균 / 모집단 표준편차

# standardScaler :
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]







