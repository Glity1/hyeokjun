import numpy as np

x1 = np.array([1,2,3]) #리스트: 벡터형태
print('x1 =', x1.shape) #데이터에 관한 형태가 출력된다. x1 = (3,) 벡터 스칼라3개짜리

x2 = np.array([[1,2,3]])
print('x2 = ', x2.shape) #x2 =  (1, 3)  행렬

x3 = np.array([[1,2],[3,4]])
print('x3 = ', x3.shape) #x3 =  (2, 2)  행렬

x4 = np.array([[1,2],[3,4],[5,6]])
print('x4 = ', x4.shape) #x4 =  (3, 2)  행렬

x5 = np.array([[[1,2],[3,4],[5,6]]])
print('x5 = ', x5.shape) #x5 =  (1, 3, 2)   행렬

x6 = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print('x6 = ', x6.shape) #x6 =  (2, 2, 2)   행렬

x7 = np.array([[[[1,2,3,4,5], [6,7,8,9,0]]]])
print('x7 = ', x7.shape) #x7 =  (1, 1, 2, 5)   행렬

x8 = np.array([[1,2,3],[4,5,6]])
print('x8 = ', x8.shape) #x8 =  (2, 3)   행렬

x9 = np.array([[[[1]]],[[[2]]]])
print('x9 = ', x9.shape) #x9 =  (2, 1, 1, 1)   행렬

x10 = np.array([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])
print('x10 = ', x10.shape) #x10 =  (2, 2, 2, 2)   행렬






















