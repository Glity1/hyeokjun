import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 저장된 데이터 불러오기
x_train = np.load('./_save/final_x_train.npy')
x_val = np.load('./_save/final_x_val.npy')
y_train = np.load('./_save/final_y_train.npy')
y_val = np.load('./_save/final_y_val.npy')

# 2. DNN 모델 구성
model = Sequential([
    Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 3. 학습
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# 4. 평가
train_pred = model.predict(x_train).reshape(-1)
val_pred = model.predict(x_val).reshape(-1)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)

print(f"✅ (Final DNN) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Final DNN) Val   RMSE: {val_rmse:.4f}")

# 5. 모델과 예측 결과 저장
model.save('./_save/final_dnn_model.h5')
np.save('./_save/final_train_pred.npy', train_pred)
np.save('./_save/final_val_pred.npy', val_pred)

# 6. 학습 곡선 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./_save/final_loss_curve.png')
plt.show()
