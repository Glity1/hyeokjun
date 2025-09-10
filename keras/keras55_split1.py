import numpy as np

a = np.array(range(1, 10))
timesteps = 3

print(a.shape) # (9,)

# 여기서 split_x란 어떤 함수인지 만들었음.

def split_x(dataset, timesteps):                         # (9, 3)
    aaa = []
    for i in range(len(dataset) - timesteps+1):          # i는 9-3+1 = 7 0부터 7까지 넣는다
        subset = dataset[i : (i+timesteps)]              # subset =  dataset의 [i 부터 i+3] 까지 i가 0부터 7까지 반복
        aaa.append(subset)                               # aaa의 [] 빈 리스트에 subset을 i가 0~7까지 반복된걸 넣어준다
        print(aaa)
# [array([1, 2, 3])]
# [array([1, 2, 3]), array([2, 3, 4])]
# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5])]
# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5]), array([4, 5, 6])]
# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5]), array([4, 5, 6]), array([5, 6, 7])]
# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5]), array([4, 5, 6]), array([5, 6, 7]), array([6, 7, 8])]
# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5]), array([4, 5, 6]), array([5, 6, 7]), array([6, 7, 8]), array([7, 8, 9])]
    
    return np.array(aaa)                                 # aaa를 numpy 배열로 반환받는다  즉  함수종료 0

# 여기서 내가 만든 split_x 라는 함수를 이용해서 썼음

bbb = split_x(a, timesteps=timesteps)  # 9-3+1 만큼 행이 생성됨  열은 timesteps 만큼 즉 (len(dataset) - timesteps+1, timesteps)
print(bbb)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]
#  [7 8 9]]

print(bbb.shape)                       # (7,3)


x = bbb[: , :-1]      # 각 행에서 마지막 값을 뺀 입력
y = bbb[:, -1]        # 각 행의 마지막 값만 가져와서 입력
print(x,y)
# [[1 2]
#  [2 3]
#  [3 4]
#  [4 5]
#  [5 6]
#  [6 7]
#  [7 8]] [3 4 5 6 7 8 9]

print(x.shape, y.shape) #(7, 2) (7,)