# m36_1에서 뽑은 4가지 결과로 5개의 모델 만들기

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
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
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 스케일링
x = np.concatenate([x_train, x_test], axis=0) / 255.0
y = np.concatenate([y_train, y_test], axis=0)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
    stratify=y
    )

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


print(x_train.shape) # (48000, 3072)

nums = [1, 30, 60, 99]
for i in nums:
    print(f"\n====== lda n_components = {i} ======")
    lda = LinearDiscriminantAnalysis(n_components=i)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

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
    model.add(Dense(100, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

    start = time.time()
    model.fit(x_train_lda, y_train_cat, epochs=300, batch_size=128,
              validation_split=0.2, callbacks=[es], verbose=1)
    end = time.time()

    #4. 평가, 예측
    loss, acc = model.evaluate(x_test_lda, y_test_cat, verbose=1)
    y_pred = model.predict(x_test_lda, verbose=1)
    y_pred_label = np.argmax(y_pred, axis=1)
    acc_score = accuracy_score(y_test, y_pred_label)
    print(f" acc : {acc_score:.4f} /  time : {end - start:.2f}초")