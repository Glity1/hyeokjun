list = ['a','b','c','d', 5]
print(list) #['a', 'b', 'c', 'd', 5]


for i in list:                                 # 그냥 출력 i = 0, 1, 2, 3, 4 대로  
    print(i)
    # a
    # b
    # c
    # d
    # 5
    
for index, value in enumerate(list):           # 순서대로 뽑아서 보는방법
    print(index, value)
    # 0 a
    # 1 b
    # 2 c
    # 3 d
    # 4 5
    
