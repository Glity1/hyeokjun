import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
dataset_csv.index = pd.to_datetime(dataset_csv.index)

# 입력 데이터 x, 타겟 데이터 y 생성
x = dataset_csv[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y = dataset_csv['wd (deg)']


# MinMaxScaler 적용
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# 2. 시퀀스 데이터 생성 함수 정의
def split_xy(x, y, timesteps, target_steps, stride):
    x_seq, y_seq = [], []
    for i in range(0, len(x) - timesteps - target_steps + 1, stride):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + target_steps])
    return np.array(x_seq), np.array(y_seq)

# 3. 시퀀스 구성 
timesteps = 144
target_steps = 144
stride = 1
x_seq, y_seq = split_xy(x_scaled, y.to_numpy(), timesteps, target_steps, stride)

# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=222, shuffle=True
)

# 5. 모델 구성
model = Sequential()
model.add(GRU(128, input_shape=(timesteps, x.shape[1]),return_sequences=True, activation='relu'))
model.add(GRU(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(target_steps, activation='linear'))  # 144개 시점 예측

path='./_save/keras53/'
model.save_weights(path+'keras53_weight.h5')

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/keras56/jena_best_model.hdf5', monitor='val_loss', save_best_only=True)

# 6. 학습
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2, callbacks=[es, mcp])

path='./_save/keras56_mcp/01_boston/'
model.save(path+'keras56_mcp_sa.h5')

# 7. 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("Loss:", loss)
y_pred = model.predict(x_test)

# 8. RMSE 계산
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("RMSE:", RMSE(y_test, y_pred))

# 9. 제출 파일 생성 
x_submit = x_scaled[-timesteps:].reshape(1, timesteps, x.shape[1])
y_submit_pred = model.predict(x_submit).flatten()

date_range = pd.date_range(start='2016-12-31 00:10', end='2017-01-01 00:00', freq='10min')
submission = pd.DataFrame({
    'Date Time': date_range,
    'wd (deg)': y_submit_pred
})
submission.to_csv('./_save/keras56/jena_서혁준_submit.csv', index=False)
print('✅ 제출 파일 저장 완료!')
