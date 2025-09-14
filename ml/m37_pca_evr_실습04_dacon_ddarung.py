import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes                               #당뇨병 diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
import time
from sklearn. metrics import accuracy_score, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x) #[1459 rows x 9 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
)

pca = PCA(n_components=9)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print(evr_cumsum)

# [0.99627425 0.99889616 0.99951909 0.99982149 0.99994769 0.99999625
#  0.99999992 1.         1.        ]
# exit()

value = [0.9960, 0.9990, 0.9999]

for i in value:
    nums = np.argmax(evr_cumsum >= i) + 1                     # 언제부터 원하는 값이 시작되는가?
    count = np.sum(evr_cumsum >= i)                         # 원하는 값의 개수는 총 몇인가?
    print(f"{i:.7f} 이상: {nums}번째부터, 총 {count}개")

# 0.9960000 이상: 1번째부터, 총 9개
# 0.9990000 이상: 3번째부터, 총 7개
# 0.9999000 이상: 5번째부터, 총 5개

exit()

nums = [5, 7, 9]

for i in nums:
    print(f"\n====== PCA n_components = {i} ======")
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
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
    model.add(Dense(1))
    
    #3. 컴파일, 훈련
    model.compile(loss= 'mse', optimizer='adam')

    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=20, verbose=1,
                    restore_best_weights=True,
                    )

    start = time.time()
    model.fit(x_train_pca, y_train, epochs=300, batch_size=64,
                validation_split=0.2, callbacks=[es], verbose=1)
    end = time.time()


    #4. 평가, 예측
    loss = model.evaluate(x_test_pca, y_test)
    results = model.predict(x_test_pca)
    r2 = r2_score(y_test, results)
    
    print(f" R2_score : {r2:.4f} / time : {end - start:.2f}초")