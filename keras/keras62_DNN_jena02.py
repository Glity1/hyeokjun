import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터 로드
path = './_data/jena/'
df = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# 2. Feature / Target 분리
x = df[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y_deg = df['wd (deg)']

# 3. 각도 → sin/cos 변환
def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

y_sin, y_cos = deg_to_sin_cos(y_deg)
y = np.stack([y_sin, y_cos], axis=1)

# 4. 시계열 데이터 생성 (입력 144개 → 출력 144개)
def split_xy(x, y, timesteps=144, target_steps=144):
    x_seq, y_seq = [], []
    for i in range(len(x) - timesteps - target_steps):
        x_seq.append(x[i:i+timesteps].values)
        y_seq.append(y[i+timesteps:i+timesteps+target_steps])
    return np.array(x_seq), np.array(y_seq)

x_seq, y_seq = split_xy(x, y, timesteps=144, target_steps=144)
print("x_seq.shape:", x_seq.shape)  # (samples, 144, 3)
print("y_seq.shape:", y_seq.shape)  # (samples, 144, 2)

# 5. Flatten input for DNN
x_seq = x_seq.reshape(x_seq.shape[0], -1)   # (samples, 144*3)
y_seq = y_seq.reshape(y_seq.shape[0], -1)   # (samples, 144*2)

# 6. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, random_state=42)

# 7. 모델 구성 (DNN)
model = Sequential([
    Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(288, activation='linear')  # 144*2 (sin, cos)
])

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/jena_dnn_best.h5', save_best_only=True, monitor='val_loss')

# 8. 모델 학습
model.fit(x_train, y_train, validation_split=0.2,
          epochs=100, batch_size=128, callbacks=[es, mcp])

# 9. 평가
loss = model.evaluate(x_test, y_test)
print("MSE Loss:", loss)

# 10. 예측
y_pred = model.predict(x_test)

# 11. RMSE 계산
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("RMSE:", RMSE(y_test, y_pred))

# 12. 마지막 데이터로 제출 생성
x_submit = x[-144:].values.reshape(1, -1)  # (1, 144*3)
y_submit_pred = model.predict(x_submit).reshape(144, 2)

# 13. sin/cos → 각도 복원
def sin_cos_to_deg(sin_val, cos_val):
    angle = np.rad2deg(np.arctan2(sin_val, cos_val))
    return (angle + 360) % 360

y_angle = sin_cos_to_deg(y_submit_pred[:, 0], y_submit_pred[:, 1])

# 14. 날짜 생성
date_range = pd.date_range(start='2016-12-31 00:10', periods=144, freq='10min')
submission = pd.DataFrame({'Date Time': date_range.strftime('%d.%m.%Y %H:%M:%S'),
                           'wd (deg)': y_angle})
submission.to_csv('./_save/jena_submit_dnn.csv', index=False)
print("✅ 제출 파일 저장 완료")
