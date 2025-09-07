import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

text1 = '오늘도 못생기고 영어를 디게 디게 디게  못 하는 이샥이는 재미없는 개그를 \
        마구 마구 마구 마구 하면서 딴짓을 한다.'
        
text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. 상진이는 마구 마구 딴짓을 한다. \
        재현은 못생기고 재미없는 딴짓을 한다.'

token = Tokenizer()  # 객체 = 대문자()=>클래스  // 인스턴스화
token.fit_on_texts([text1, text2])

# 단어 종류 확인
print(token.word_index)
# {'마구': 1, '디게': 2, '딴짓을': 3, '한다': 4, '오늘도': 5, '못생기고': 6, '재미없는': 7, '영어를': 8, '못': 9, '하는': 10, 
# '이샥이는': 11, '개그를': 12, '하면서': 13, '박석사가': 14, '자아를': 15, '찾아냈다': 16, '상진이는': 17, '재현은': 18}

# 각 단어가 몇번 나왔지?
print(token.word_counts)
# OrderedDict([('오늘도', 2), ('못생기고', 2), ('영어를', 1), ('디게', 5), ('못', 1), ('하는', 1), ('이샥이는', 1), ('재미없는', 2), ('개그를', 1), ('마구', 6),
# ('하면서', 1), ('딴짓을', 3), ('한다', 3), (' 박석사가', 1), ('자아를', 1), ('찾아냈다', 1), ('상진이는', 1), ('재현은', 1)])

# 각 단어에 대해 수치화 즉 머신이 읽을 수 있게 해준다.
x = token.texts_to_sequences([text1, text2])
print(x) #[[5, 6, 8, 2, 2, 2, 9, 10, 11, 7, 12, 1, 1, 1, 1, 13, 3, 4], [5, 14, 15, 2, 2, 16, 17, 1, 1, 1, 3, 4, 18, 6, 7, 3, 4]]
x = np.array(x)
print(x) #  [list([5, 6, 8, 2, 2, 2, 9, 10, 11, 7, 12, 1, 1, 1, 1, 13, 3, 4]) 
         #  list([5, 14, 15, 2, 2, 16, 17, 1, 1, 1, 3, 4, 18, 6, 7, 3, 4])
print(x.shape) # (2,)
x = np.concatenate(x)
print(x) # [ 5  6  8  2  2  2  9 10 11  7 12  1  1  1  1 13  3  4  5 14 15  2  2 16 17  1  1  1  3  4 18  6  7  3  4]
print(x.shape) # (34,)

# exit()
# 리스트안에는 다양한 종류의 데이터가 들어갈 수 있다 (수치, 문자)처럼

####### 원핫 3가지 만들것 ########   0이 너무 많은게 문제

#1. pandas

x1 = pd.get_dummies(x)
print(x1)
print(x1.shape)  

#2. sklearn

from sklearn.preprocessing import OneHotEncoder 
x_arr = x.reshape(-1, 1)     # numpy 배열로 바꿔줘야 reshape이 가능함

ohe = OneHotEncoder(sparse=False)      # reshape을 하는 이유 ohe는 2차원 배열만 받는다.
x2 = ohe.fit_transform(x_arr)
print(x2)
print("Shape:", x2.shape)


#3. keras

from tensorflow.keras.utils import to_categorical    
x_reshaped = x.reshape(1, -1)
x3 = to_categorical(x_reshaped)
x3 = x3[:, :, 1:]
print(x3)
print("Shape:", x3.shape)