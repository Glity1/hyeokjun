from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,
    # maxlen=100,
)

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(np.unique(y_train, return_counts=True)) # [0, 1], [12500, 12500]

print("영화평의 최대 길이 : ", max(len(x) for x in x_train)) # 2494
print("영화평의 최소 길이 : ", min(len(x) for x in x_train)) # 11
print("영화평의 평균 길이 : ", sum(map(len, x_train)) / len(x_train)) # 238.71364

sequence_len = 500

x_train = pad_sequences(
    x_train,
    maxlen=sequence_len,
    padding='pre',
    truncating='pre'
)

x_test = pad_sequences(
    x_test,
    maxlen=sequence_len,
    padding='pre',
    truncating='pre'   
)

# 모델
model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Embedding(input_dim=10001, output_dim=1000, input_length=sequence_len))
model.add(LSTM(units=100, activation='tanh'))
model.add(Dense(units=60, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 컴파일, 학습
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es]
)

# 평가, 예측
y_pred = (model.predict(x_test) >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy : {acc:.4f}")
