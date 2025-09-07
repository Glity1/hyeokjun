import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # ReduceLROnPlateau 추가
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator # ImageDataGenerator 임포트

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Original x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"Original x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# 이미지 픽셀 값을 0-1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# y 데이터를 1차원 배열로 변환 후 원-핫 인코딩
y_train = y_train.reshape(-1,) # (50000, 1) -> (50000,)
y_test = y_test.reshape(-1,)   # (10000, 1) -> (10000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(f"After one-hot encoding y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# 데이터 증강 설정
# 훈련 데이터에 다양한 변형을 적용하여 모델의 일반화 성능 향상
datagen = ImageDataGenerator(
    rotation_range=15,          # -15 ~ +15도 범위 내에서 무작위 회전
    width_shift_range=0.1,      # 이미지 너비의 10% 내외로 좌우 이동
    height_shift_range=0.1,     # 이미지 높이의 10% 내외로 상하 이동
    horizontal_flip=True,       # 무작위 수평 뒤집기
    zoom_range=0.1,             # 10% 내외로 무작위 확대/축소
    # fill_mode='nearest'       # 변환 후 빈 공간을 채우는 방식 (기본값: 'nearest')
)

# 데이터 증강 파이프라인에 훈련 데이터를 맞춥니다.
# featurewise_center나 featurewise_std_normalization을 사용할 때 필요하지만,
# 여기서는 0-1 정규화를 이미 했으므로 필수는 아닙니다.
# datagen.fit(x_train)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) # pool_size=(2,2)는 기본값이므로 생략 가능
model.add(Dropout(0.5))

model.add(Conv2D(128, (3,3), padding='same')) # 필터 수 증가
model.add(BatchNormalization()) # BatchNormalization 추가
model.add(Activation('relu')) 
model.add(Conv2D(128, (3,3), padding='same')) # 필터 수 증가
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35)) # Dropout 비율 조정

model.add(Flatten())
model.add(Dense(units=256)) # Dense Layer 유닛 수 증가
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.2)) # Dropout 비율 조정 (0.1 -> 0.2)
model.add(Dense(units=10, activation='softmax'))

# 모델 요약 (선택 사항)
model.summary()

# 3. 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# EarlyStopping 콜백 설정
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20, # 20 에포크 동안 val_loss 개선 없으면 중단
    verbose=1,
    restore_best_weights=True # 가장 성능이 좋았던 가중치 복원
)

# 학습률 감소 콜백 추가 (ReduceLROnPlateau)
# val_loss가 개선되지 않으면 학습률을 0.1배로 감소시킵니다.
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1, # 학습률을 0.1배로 줄임
    patience=10, # 10 에포크 동안 val_loss 개선 없으면 학습률 감소
    verbose=1,
    min_lr=0.00001 # 최소 학습률 설정
)

# 훈련 데이터에 데이터 증강 적용
# validation_split 대신 validation_data를 사용하여 x_test, y_test를 검증셋으로 사용
# ImageDataGenerator.flow()는 훈련 데이터만 증강하고, 검증 데이터는 원본 그대로 사용합니다.
hist = model.fit(
    datagen.flow(x_train, y_train, batch_size=64), # 데이터 증강된 훈련 데이터 사용
    epochs=50,
    verbose=1,
    validation_data=(x_test, y_test), # 테스트 셋을 검증 셋으로 활용 (증강X)
    callbacks=[es, lr_scheduler] # EarlyStopping과 ReduceLROnPlateau 콜백 적용
)

# 4. 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {loss[0]:.4f}')
print(f'Test accuracy: {loss[1]:.4f}')

y_pred = model.predict(x_test)
y_test_labels = np.argmax(y_test.values, axis=1) # y_test는 이미 원-핫 인코딩되어 있으므로 .values 필요
y_pred_labels = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Accuracy score: {acc:.4f}")

# 선택 사항: 훈련 과정 시각화 (손실 및 정확도)
plt.figure(figsize=(12, 5))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], label='Train Accuracy')
plt.plot(hist.history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()