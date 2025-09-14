# 결측치 처리할 때 많이 쓴다.

import numpy as np
data = [10,20,30,40,50]

print(np.percentile(data, 25))  # 17.5

data = [10,20,30,40]           # index는 0, 1, 2, 3 이다.

print(np.percentile(data, 25))  # 17.5

"""

index의 위치 찾기
rank = (n - 1) * (q / 100)
     = (4 - 1) * (25 / 100)
     = 3 * 0.25 = 0.75            # index 위치가 0.75



# 보간법
# 작은값 = data의 0번째 = 10
# 큰값 = data의 1번째 = 20

# 백분위값 = 작은값 + (큰값 - 작은값) * rank
          = 10 + (20 - 10) * 0.75
          = 10 + 7.5 = 17.5 

"""

