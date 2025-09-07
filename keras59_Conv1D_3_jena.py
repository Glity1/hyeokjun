import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Dropout, Bidirectional, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# sin/cos 변환 함수 정의
def sin_cos_transform(degree):
    radians = np.deg2rad(degree)
    return np.sin(radians), np.cos(radians)

# 입력 데이터 x, 타겟 데이터 y 생성
x = dataset_csv[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y_raw = dataset_csv['wd (deg)']                                         # wd(deg) 컬럼만 y_raw에 넣어주겠다 // y_raw는 풍향을 나타내는 1차원 시리즈 (벡터)
y_sin, y_cos = sin_cos_transform(y_raw)                                 # 풍향 각도(도 단위)를 사인(sin), 코사인(cos) 값으로 각각 변환해준다.
y = np.stack((y_sin, y_cos), axis=1)                                    # y_sin, y_cos 을 나란히 쌓아서 (N, 2) 배열 만들기 (y_sin , y_cos)
                                                                        # y_sin, 즉 sin(풍향) 값 (Y축 방향)
                                                                        # y_cos, 즉 cos(풍향) 값 (X축 방향)

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
stride = 144
x_seq, y_seq = split_xy(x, y, timesteps, target_steps, stride)

print(x_seq.shape, y_seq.shape) #(2919, 144, 3) (2919, 144, 2)

# exit()
# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=777, shuffle=True
)

# 5. 모델 구성
# model = Sequential([
#     GRU(128, input_shape=(timesteps, x.shape[1]), return_sequences=True, activation='relu'),
#     Bidirectional(GRU(32)),
#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(target_steps * 2, activation='linear'),  # sin, cos 각각 144개
# ])

model = Sequential([
    Conv1D(filters=128, kernel_size=2, input_shape=(timesteps, x.shape[1]), padding='same',activation='relu'),
    Conv1D(filters=64, kernel_size=2, input_shape=(timesteps, x.shape[1]), padding='same',activation='relu'),
    Conv1D(filters=32, kernel_size=2, input_shape=(timesteps, x.shape[1]), activation='relu'),
    GRU(64, input_shape=(timesteps, x.shape[1]), return_sequences=True, activation='relu'),
    Bidirectional(GRU(32)),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(target_steps * 2, activation='linear'),  # sin, cos 각각 144개
])
# exit()
model.compile(loss='mse', optimizer='adam')

# 6. 콜백
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/keras56/jena_best_model2.hdf5', monitor='val_loss', save_best_only=True)

# 7. 학습
model.fit(x_train, y_train.reshape(y_train.shape[0], -1),
          epochs=100, batch_size=128, validation_split=0.2,
          callbacks=[es, mcp])

model.save('./_save/keras56/jena_final_model2.h5')
model.save_weights('./_save/keras56/jena_final_weights2.h5')

# 8. 평가 및 예측
loss = model.evaluate(x_test, y_test.reshape(y_test.shape[0], -1))
print("Loss:", loss)
y_pred = model.predict(x_test)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1, 2), y_pred.reshape(-1, 2)))

print("RMSE:", RMSE(y_test, y_pred.reshape(-1, 144, 2)))

# 9. 제출 파일 생성
x_submit = np.array(x[-timesteps:]).reshape(1, timesteps, x.shape[1])
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
    'wd (deg)': y_submit_angle
    })
submission.to_csv('./_save/keras56/jena_서혁준_submit24.csv', index=False)
print('✅ 제출 파일 저장 완료!')


# [144 rows x 1 columns]
# RMSE: 53.75661678594249

# Conv1D 4개층
# Loss: 0.44774240255355835
# RMSE: 0.669135555829332

