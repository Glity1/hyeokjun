# 65-1 DNN1 copy

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM        # 레이어상에서 Embedding 해준다
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# 데이터
docs = ['너무 재미있다', '참 최고에요', '참 잘만든 영화예요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', 
        '글쎄', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다.',
        '참 재밌네요.', '석준이 바보', '준희 잘생겼다', '이삭이 또 구라친다']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

# 패딩
pad_x = pad_sequences(x, padding='pre', maxlen=6)

# print(pad_x)
print(pad_x.shape) #(15, 5)

# OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# x_ohe = ohe.fit_transform(pad_x)

# 훈련/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(pad_x, labels, test_size=0.2, random_state=42)

#2. 모델구성  Embedding에 들어간 파라미터에 대해 알아볼것.
#2-1. Embedding
# model = Sequential([
#         Embedding(input_dim=31, output_dim=100, input_length=5),   # input_dim = 단어사전의 개수(말뭉치의 개수) : 10이면  0 ~ 9 사이의 정수 인덱스를 가진 단어들이 임베딩 가능하다는 의미
#         LSTM(16),                                                  # output_dim = unit => 출력의 개수 : 다음 레이어로 전달하는 노드의 개수 (변경 가능) / 각 단어가 임베딩된 후의 벡터 차원 수입니다. /각 단어는 5차원 실수 벡터로 변환됩니다.예를 들어, 단어 0 → [0.1, -0.3, 0.9, 0.5, -0.2]
#         Dense(1, activation='sigmoid')                             # input_length = 컬럼의 개수, 모델에 입력되는 문장의 시퀀스의 길이(개수)입니다. /각 입력 샘플이 3개의 단어(정수 인덱스)로 구성된다는 의미입니다예: [1, 6, 3], [2, 4, 0] 같은 형태
# ])                                                                

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100        # Embedding 파라미터 수 = input_dim × output_dim

#  lstm (LSTM)                 (None, 16)                7488        # LSTM 파라미터 수 = 4 × [(input_dim × units) + (units × units) + units]

#  dense (Dense)               (None, 1)                 17

# =================================================================
# Total params: 10,605
# Trainable params: 10,605
# Non-trainable params: 0
# _________________________________________________________________

#2-2. Embedding
# model = Sequential([
#         Embedding(input_dim=31, output_dim=100,),                    # length를 따로 명시안해도 알아서 맞춰준다 // 알면 제대로 넣고 모르면 그냥 적지마라.
#         LSTM(16),                                                 
#         Dense(1, activation='sigmoid')           
# ]) 
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100      

#  lstm (LSTM)                 (None, 16)                7488      

#  dense (Dense)               (None, 1)                 17        

# =================================================================
# Total params: 10,605
# Trainable params: 10,605
# Non-trainable params: 0
# _________________________________________________________________


#2-3. Embedding
# model = Sequential([
#         Embedding(input_dim=13, output_dim=100,),                    # input_dim 적으면 단어사전을 줄여버리니까 성능 저하가 되서 크면 메모리가 많이 할당된다. 둘다 진행은 된다 
#         LSTM(16),                                                 
#         Dense(1, activation='sigmoid')           
# ]) 

#2-4. Embedding
model = Sequential([
        # Embedding(31, 100,),
        # Embedding(31, 100, 5),                                     #   진행 불가 
        # Embedding(31, 100, input_length=5),                        #  input_length 는 명시해줘야 한다 안적고 숫자만 넣으면 안 돌아간다 = column의 개수
        Embedding(31, 100, input_length=1),                          #  1은 먹힌다 none이랑 다를바가 없음          
        LSTM(16),                                                 
        Dense(1, activation='sigmoid')           
]) 
# model.summary()


# exit()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

# 평가
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")


# 예측할 문장
x_pred = (['이삭이 참 잘생겼다'])
x_pred = token.texts_to_sequences(x_pred)
x_pred = pad_sequences(x, padding='pre', maxlen=5)

# 예측
y_pred = model.predict(x_pred)
print("결과:", "긍정" if y_pred[0][0] > 0.5 else "부정")
print('이삭이 참 잘생겼다 의 결과 : ', np.round(y_pred))