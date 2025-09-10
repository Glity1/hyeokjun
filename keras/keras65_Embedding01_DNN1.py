import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
pad_x = pad_sequences(x, padding='pre', maxlen=5)

# OneHotEncoder
ohe = OneHotEncoder(sparse=False)
x_ohe = ohe.fit_transform(pad_x)

# 훈련/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(x_ohe, labels, test_size=0.2, random_state=42)

# 모델
model = Sequential()
model.add(Dense(32, input_shape=(x_ohe.shape[1],), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=0)

# 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")


# 예측할 문장
new_text = ['이삭이 참 잘생겼다']

# 토큰화 → 시퀀스 변환 → 패딩 → OHE
new_seq = token.texts_to_sequences(new_text)
new_pad = pad_sequences(new_seq, padding='pre', maxlen=5)
new_ohe = ohe.transform(new_pad)

# 예측
pred = model.predict(new_ohe)
print("예측 확률:", pred[0][0])
print("결과:", "긍정" if pred[0][0] > 0.5 else "부정")