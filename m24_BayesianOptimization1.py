param_bouns = {'x1' : (-1, 5),    # param_bouns : 파라미터의 범위 // # x1 라는 키에는 (-1 부터 5) 튜플형태
               'x2' : (0, 4)}     # x2 라는 키에는 (0 부터 4) value가 들어가있다.

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 + 10       # **2 : 제곱한다.   -x1을 제곱할려고 하면 (-x1)**2으로 해줘야함.

# y = -x1의 제곱 -(x2-2)의 제곱 + 10  y의 최댓값은 10을 넘을 수 없다 // x1 = 0 x2= 2 일 때 최댓값
# y 최대값을 구하기 위한 함수

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,                   # 함수는 y_function을 쓴다.  // 블랙박스 함수
    pbounds=param_bouns,
    random_state=333,
)

optimizer.maximize(init_points=5,     # 초기 훈련 5번     # 총 25번 돌려라
                   n_iter=30)         # 반복 훈련 20번

print(optimizer.max)

