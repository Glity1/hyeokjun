import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

text = '오늘도 영어를 되게 못 하는 이샥이는 재미없는 개그를 \
        마구 마구 마구 하면서 딴 짓을 한다.'
"""

음절 : 오, 늘, 도 
어절 : 띄어쓰는 부분 : 오늘도
형태소 : 뜻을 가진 최소단위 : 오늘 , 도

어절 단위로 수치화 한다.
label Encoding

"""

token = Tokenizer()  # 객체 = 대문자()=>클래스  // 인스턴스화
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '오늘도': 2, '영어를': 3, '되게': 4, '못': 5, '하는': 6, '이샥이는': 7, '재미없는': 8, '개그를': 9, '하면서': 10, '딴': 11, '짓을': 12, '한다': 13} 단어종류의 개수 14개 0포함

print(token.word_counts)
# OrderedDict([('오늘도', 1), ('영어를', 1), ('되게', 1), ('못', 1), ('하는', 1), ('이샥이는', 1), ('재미없는', 1), ('개그를', 1), ('마구', 2), ('하면서', 1), ('딴', 1), ('짓을', 1), ('한다', 1)])
# 각단어가 몇번 나왔는지

x = token.texts_to_sequences([text])[0]
x = np.array(x) 
print(x) #[[2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11, 12, 13]]
print(x.shape) #(15,)

####### 원핫 3가지 만들것 ########

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


