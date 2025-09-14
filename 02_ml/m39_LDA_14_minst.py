# m36_1에서 뽑은 4가지 결과로 5개의 모델 만들기

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn. metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 스케일링
x = np.concatenate([x_train, x_test], axis=0) / 255.0
y = np.concatenate([y_train, y_test], axis=0)
x = x.reshape(70000, 784)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
    stratify=y
    )

print(x_train.shape) #(56000, 784)
 # (784, 10-1) 만큼 가능

nums = [1, 3, 9]
for i in nums:
    print(f"\n====== lda n_components = {i} ======")
    lda = LinearDiscriminantAnalysis(n_components=i)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    #2. 모델
    model = Sequential()
    model.add(Dense(128, input_dim=i, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

    start = time.time()
    model.fit(x_train_lda, y_train_cat, epochs=100, batch_size=64,
              validation_split=0.2, callbacks=[es], verbose=0)
    end = time.time()

    #4. 평가, 예측
    loss, acc = model.evaluate(x_test_lda, y_test_cat, verbose=0)
    y_pred = model.predict(x_test_lda, verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)
    acc_score = accuracy_score(y_test, y_pred_label)
    print(f" acc : {acc_score:.4f} /  time : {end - start:.2f}초")