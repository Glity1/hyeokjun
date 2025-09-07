"""
jena의 rnn의 shape를 가져와서 단순 reshape 한다.

(N, timesteps, feature)  3차원
->
(N, timesteps * feature) 2차원으로 단순 reshape해서 해볼것.

"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# dataset_csv.index = pd.to_datetime(dataset_csv.index)

# # 2. 원래 있어야 할 시간 범위 생성 (10분 단위)
# expected_index = pd.date_range(start=dataset_csv.index.min(),
#                                end=dataset_csv.index.max(),10.08.2009
#                                freq='10min')

# # 3. 누락된 시간 확인
# missing_times = expected_index.difference(dataset_csv.index)

# # 4. 결과 출력
# print("누락된 시간 개수:", len(missing_times))  # 누락된 시간 개수: 544
# print("누락된 시간 목록:\n", missing_times)

# 중복된 인덱스 확인
# duplicates = dataset_csv.index[dataset_csv.index.duplicated()]
# print("중복된 인덱스 개수:", len(duplicates))     # 327개
# print("중복된 인덱스 값:\n", duplicates.unique())

# 1. 중복 제거 전 총 개수
# print("중복 제거 전 개수:", len(dataset_csv))   #420551

# # 2. 중복 제거 수행
# dataset_csv = dataset_csv[~dataset_csv.index.duplicated(keep='first')]

# # 3. 중복 제거 후 총 개수
# print("중복 제거 후 개수:", len(dataset_csv))  #420224

# print("중복된 인덱스 개수:", dataset_csv.index.duplicated().sum())   # 중복된 인덱스 개수: 0

# 1. 데이터 인덱스를 datetime으로 변환 (기존 인덱스가 문자열일 경우)
# dataset_csv.index = pd.to_datetime(dataset_csv.index, dayfirst=True)  # dayfirst=True → 일.월.년 처리

# # 2. 실제 존재하는 시간: '일.월.년 시:분' 형태 문자열
# actual_index = dataset_csv.index.strftime('%d.%m.%Y %H:%M')

# # 3. 기대되는 전체 시간 범위 (10분 간격)
# expected_range = pd.date_range(
#     start=dataset_csv.index.min(),
#     end=dataset_csv.index.max(),
#     freq='10min'
# )
# expected_index = expected_range.strftime('%d.%m.%Y %H:%M')

# # 4. 누락된 시간 찾기
# missing_times = sorted(set(expected_index) - set(actual_index))

# # 5. 출력
# print("📌 누락된 10분 단위 시간 개수:", len(missing_times))
# print("🔍 누락된 시간 목록:")
# for t in missing_times:
#     print(t)

# exit()

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


# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=777, shuffle=True
)

x_train = x_train.reshape(-1, 432) 
x_test = x_test.reshape(-1, 432)  
y_train = y_train.reshape(-1, 288)
y_test = y_test.reshape(-1, 288)

print(x_train.shape, x_test.shape)
# exit()

# 5. 모델 구성
# model = Sequential([
#     GRU(128, input_shape=(timesteps, x.shape[1]), return_sequences=True, activation='relu'),
#     Bidirectional(GRU(32)),
#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(target_steps * 2, activation='linear'),  # sin, cos 각각 144개
# ])

model=Sequential()
model.add(Dense(10, input_dim=432))
model.add(Dropout(0.1))
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(timesteps*2, activation='linear'))

# exit()

model.compile(loss='mse', optimizer='adam')

# 6. 콜백
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/keras56/jena_best_model.hdf5', monitor='val_loss', save_best_only=True)

# 7. 학습
model.fit(x_train, y_train,
          epochs=30, batch_size=128, validation_split=0.2,
          callbacks=[es, mcp])

model.save('./_save/keras56/jena_final_model2.h5')
model.save_weights('./_save/keras56/jena_final_weights2.h5')

# 8. 평가 및 예측
loss = model.evaluate(x_test, y_test.reshape(y_test.shape[0], -1))
print("Loss:", loss)
y_pred = model.predict(x_test)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.reshape(-1, 2), y_pred.reshape(-1, 2)))

print("RMSE:", RMSE(y_test, y_pred.reshape(-1, 2)))

# 9. 제출 파일 생성
x_submit = np.array(x[-timesteps:]).reshape(1, timesteps*x.shape[1])
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
submission.to_csv('./_save/keras56/jena_서혁준_submit232.csv', index=False)
print('✅ 제출 파일 저장 완료!')


# [144 rows x 1 columns]
# RMSE: 53.75661678594249