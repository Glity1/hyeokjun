# keras 21-2 copy
#https://www.kaggle.com/competitions/playground-series-s4e1/submissions

import numpy as np                                           
import pandas as pd                                                                               
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt        
import matplotlib.font_manager as fm
import matplotlib as mpl
from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTENC

# 1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자형 데이터 수치화
le = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

# print(train_csv.info())
# print(train_csv.describe())

# exit()
# 불필요한 col 제거
train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

# col 분리
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

smotenc = SMOTENC(random_state=337, 
                  categorical_features= [2,3,6,7,8])
x_res, y_res = smotenc.fit_resample(x, y)

# print(x_res)
# exit()

print(y.value_counts())
 # 0    130113
 # 1     34921

# 데이터 스케일링
scaler = StandardScaler()
x_res = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_res, y_res,
    test_size=0.2,
    random_state=8141,
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(256, input_dim=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=256,
    validation_split=0.2,
    callbacks=[es],
    verbose=1,
)

# 4. 평가
results = model.evaluate(x_test, y_test, verbose=0)
print("loss : ", results[0])
print("accuracy : ", round(results[1], 5))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

roc_auc = roc_auc_score(y_test, y_predict)
print("ROC_AUC Score : ", roc_auc)

# 5. 제출 파일 생성
y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_1550.csv', index=True)

# 6. 학습 과정 시각화
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.grid()
plt.legend()
plt.title('Loss 및 Val_loss')
plt.show()
