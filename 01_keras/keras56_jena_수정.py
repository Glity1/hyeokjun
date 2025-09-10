import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
dataset_csv.index = pd.to_datetime(dataset_csv.index)

# sin/cos 변환 함수 정의
def sin_cos_transform(degree):
    radians = np.deg2rad(degree)
    return np.sin(radians), np.cos(radians)

# 입력 데이터 x, 타겟 데이터 y 생성
x = dataset_csv[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y_raw = dataset_csv['wd (deg)']
y_sin, y_cos = sin_cos_transform(y_raw)
y = np.stack((y_sin, y_cos), axis=1)

# MinMaxScaler 적용 (rh 제외했으므로 적용 가능)
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
x_seq, y_seq = split_xy(x_scaled, y, timesteps, target_steps, stride)

# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=222, shuffle=True
)

# 5. 모델 구성
model = Sequential()
model.add(GRU(128, input_shape=(timesteps, x.shape[1]), return_sequences=True, activation='relu'))
model.add(GRU(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(target_steps * 2, activation='linear'))  # sin, cos 각각 144개
model.compile(loss='mse', optimizer='adam')

# 6. 콜백
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/keras56/jena_best_model.hdf5', monitor='val_loss', save_best_only=True)

# 7. 학습
model.fit(x_train, y_train.reshape(y_train.shape[0], -1),
          epochs=10, batch_size=128, validation_split=0.2,
          callbacks=[es, mcp])

model.save('./_save/keras56/jena_final_model.h5')
model.save_weights('./_save/keras56/jena_final_weights.h5')

# 8. 평가 및 예측
loss = model.evaluate(x_test, y_test.reshape(y_test.shape[0], -1))
print("Loss:", loss)
y_pred = model.predict(x_test)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1, 2), y_pred.reshape(-1, 2)))

print("RMSE:", RMSE(y_test, y_pred.reshape(-1, 144, 2)))

# 9. 제출 파일 생성
x_submit = x_scaled[-timesteps:].reshape(1, timesteps, x.shape[1])
y_submit_pred = model.predict(x_submit).reshape(144, 2)

# sin, cos -> 각도로 변환
y_submit_angle = np.rad2deg(np.arctan2(y_submit_pred[:, 0], y_submit_pred[:, 1]))
y_submit_angle = (y_submit_angle + 360) % 360  # 0~360도 범위 보정

# 저장
date_range = pd.date_range(start='2016-12-31 00:10', end='2017-01-01 00:00', freq='10min')
formatted_date_range = date_range.strftime('%d.%m.%Y %H:%M:%S')

# 제출 파일 생성
submission = pd.DataFrame({
    'Date Time': formatted_date_range,
    'wd (deg)': y_submit_pred
    })
submission.to_csv('./_save/keras56/jena_서혁준_submit.csv', index=False)
print('✅ 제출 파일 저장 완료!')
