# 1. 라이브러리
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 2. 저장된 데이터 불러오기
x_train = np.load('./_save/final_x_train.npy')
x_val = np.load('./_save/final_x_val.npy')
y_train = np.load('./_save/final_y_train.npy')
y_val = np.load('./_save/final_y_val.npy')

# 3. DNN 모델 구성
dnn = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
dnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = dnn.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# 4. XGBoost 모델 학습
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(x_train, y_train)

# 5. 예측 및 앙상블
pred_dnn_train = dnn.predict(x_train).reshape(-1)
pred_dnn_val = dnn.predict(x_val).reshape(-1)
pred_xgb_train = xgb.predict(x_train)
pred_xgb_val = xgb.predict(x_val)

# Soft Voting: DNN 0.7 + XGB 0.3
train_pred = 0.7 * pred_dnn_train + 0.3 * pred_xgb_train
val_pred = 0.7 * pred_dnn_val + 0.3 * pred_xgb_val

train_rmse = mean_squared_error(y_train, train_pred, squared=False)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)

print(f"✅ (Ensemble) Train RMSE: {train_rmse:.4f}")
print(f"✅ (Ensemble) Val   RMSE: {val_rmse:.4f}")

# 6. 모델 저장
dnn.save('./_save/final_dnn_model.h5')
joblib.dump(xgb, './_save/final_xgb_model.pkl')

# 7. 시각화 (1) 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('DNN 학습 곡선')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/final_learning_curve.png')
plt.show()

# 8. 시각화 (2) 예측 vs 실제
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('검증 데이터: 예측 vs 실제')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_save/final_prediction_vs_actual.png')
plt.show()
