from tensorflow.keras.datasets import reuters  # 뉴스
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM        # 레이어상에서 Embedding 해준다
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# 다중분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=1000, # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개를 뽑는다.
        test_split=0.2,
        # maxlen=200,     # 최대 단어 길이가 200개까지 있는 문장으로
        )

# print(x_train) 
# [list([1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 2, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 
# 7, 48, 9, 2, 2, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 2, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12])
#  list([1, 2, 699, 2, 2, 56, 2, 2, 9, 56, 2, 2, 81, 5, 2, 57, 366, 737, 132, 20, 2, 7, 2, 49, 2, 2, 2, 2, 699, 2, 8, 7, 10, 241, 16, 855, 129, 231, 783, 5, 4, 587, 2, 2, 2, 775, 7, 48, 34, 191, 44, 35, 2, 505, 17, 12]) 
#  list([1, 178, 53, 321, 26, 14, 948, 26, 178, 39, 44, 2, 2, 14, 191, 59, 11, 86, 539, 63, 11, 14, 892, 61, 11, 123, 197, 2, 258, 44, 11, 15, 58, 462, 26, 53, 14, 597, 61, 11, 15, 58, 19, 942, 15, 53, 105, 39, 633, 472, 927, 53, 46, 22, 710, 220, 851, 2, 9, 2, 282, 5, 317, 65, 9, 659, 249, 2, 196, 47, 11, 428, 410, 61, 59, 20, 22, 10, 29, 254, 17, 12])
#  ...
#  list([1, 361, 372, 8, 77, 62, 325, 2, 336, 5, 2, 37, 412, 453, 2, 229, 334, 13, 4, 867, 76, 4, 76, 2, 6, 264, 2, 18, 82, 95, 97, 2, 4, 2, 649, 18, 82, 554, 136, 4, 143, 334, 290, 126, 5, 4, 2, 777, 2, 2, 13, 954, 7, 4, 314, 912, 224, 4, 2, 2, 54, 429, 2, 18, 82, 5, 496, 2, 229, 57, 85, 385, 593, 6, 4, 867, 76, 17, 12])
#  list([1, 2, 71, 8, 16, 385, 4, 611, 2, 9, 608, 100, 280, 5, 126, 5, 25, 355, 873, 220, 6, 2, 2, 9, 2, 71, 10, 2, 381, 49, 2, 7, 2, 73, 2, 316, 62, 45, 889, 2, 355, 873, 220, 55, 267, 7, 2, 687, 632, 9, 40, 303, 88, 5, 69, 191, 11, 15, 17, 12])
#  list([1, 245, 273, 110, 156, 53, 272, 26, 14, 158, 26, 39, 2, 2, 14, 2, 2, 86, 32, 2, 2, 14, 19, 2, 2, 17, 12])]
# print(x_train.shape, x_test.shape)      # (8982,) (2246,)
# print(y_train.shape, y_test.shape)      # (8982,) (2246,)
# print(y_train[0])                  # 3
# print(y_train)                  # [ 3  4  3 ... 25  3 25]

# y_train = y_train.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y_ohe = ohe.fit_transform(y_train)

# print(y_ohe.shape)  #(8982, 46)
# exit()

# print(type(x_train))
# print(type(x_train[0]))

# print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #뉴스기사의 최대길이 :  2376
# print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train)) #뉴스기사의 최소길이 :  13
# print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train)) #뉴스기사의 평균길이 : 145.5398574927633
                               # map : 반복문 없이 for문처럼 작동하는 함수
# exit()
# np.unique()는 NumPy 배열에서 중복을 제거하고, 고유한 값들만 정렬된 상태로 반환해주는 함수
print(np.unique(y_train))   # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]  
                            # 총 46개다 0~45까지
pad_x = pad_sequences(x_train, padding='pre', maxlen=100)
print(pad_x.shape) #(8982,100)

max_value = np.max(pad_x)
print("pad_x에서 가장 큰 값:", max_value) # 999

# exit()

#2. 모델구성
model = Sequential([
        Embedding(input_dim=1000, output_dim=16,),                    # input_dim 적으면 단어사전을 줄여버리니까 성능 저하가 되서 크면 메모리가 많이 할당된다. 둘다 진행은 된다 
        LSTM(16),                     
        Dense(64),           
        Dense(32),                                      
        Dense(46, activation='softmax')           
]) 

#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(pad_x, y_train, epochs=100, batch_size=32, verbose =1)

#4. 평가, 예측
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=100)
loss, acc = model.evaluate(pad_x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Test Accuracy: 0.0467