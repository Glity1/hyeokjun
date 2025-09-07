import numpy as np                                           
import pandas as pd                                                                               
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder #StandardScaler
import matplotlib.pyplot as plt        
import matplotlib.font_manager as fm
import matplotlib as mpl
from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler

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

# 불필요한 col 제거
train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

# col 분리
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# 2차 col 분리
# binary_cols = ['Gender', 'Is Active Member','Has Cr Card']
# continuous_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
# print(y.value_counts())
#  0    130113
#  1     34921

# 데이터 스케일링
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# test_csv = scaler.transform(test_csv)

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=31,
)

# 데이터 스케일링

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)
# scaler = MinMaxScaler()
# scaler.fit(x_train[continuous_cols])

# x_train_scaled = x_train.copy()
# x_test_scaled = x_test.copy()
# test_csv_scaled = test_csv.copy()

# x_train_scaled[continuous_cols] = scaler.transform(x_train[continuous_cols])
# x_test_scaled[continuous_cols] = scaler.transform(x_test[continuous_cols])
# test_csv_scaled[continuous_cols] = scaler.transform(test_csv[continuous_cols])

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0], x_test_scaled.shape[1], 1)



# 2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(10, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# path='./_save/keras28_mcp/08_kaggle_bank/'
# model.save_weights(path+'keras28_MCP_save_08_kaggle_bank.h5')

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

path='./_save/keras28_mcp/08_kaggle_bank/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras33_mcp_08_kaggle_bank.hdf5'
    )

hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es, mcp],
    verbose=1,
    class_weight = class_weight_dict
)

path='./_save/keras28_mcp/08_kaggle_bank/'
model.save(path+'keras33_mcp_save_08_kaggle_bank.h5')

# 4. 평가
results = model.evaluate(x_test_scaled, y_test, verbose=0)
print("loss : ", results[0])
print("accuracy : ", round(results[1], 5))

test_csv = np.array(test_csv).reshape(test_csv.shape[0], test_csv.shape[1], 1)
y_predict = model.predict(x_test_scaled)
y_predict = np.round(y_predict)

acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

roc_auc = roc_auc_score(y_test, y_predict)
print("ROC_AUC Score : ", roc_auc)

# 5. 제출 파일 생성
test_csv_scaled = np.array(test_csv_scaled).reshape(test_csv_scaled.shape[0], test_csv_scaled.shape[1], 1)
y_submit = model.predict(test_csv_scaled)
y_submit = np.round(y_submit)

submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_1200.csv', index=True)

# 6. 그래프로 시각화
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

"""

loss :  0.6852115392684937
accuracy :  0.78995
acc_score :  0.7899536461962614
ROC_AUC Score :  0.5

dropout 변경 후

loss :  0.6355645656585693
accuracy :  0.78995
acc_score :  0.7899536461962614
ROC_AUC Score :  0.5

###########
loss :  0.6884922981262207
accuracy :  0.78995
acc_score :  0.7899536461962614
ROC_AUC Score :  0.5

"""
