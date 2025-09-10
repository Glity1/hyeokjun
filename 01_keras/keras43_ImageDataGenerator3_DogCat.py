# C:\study25ju\_data\kaggle\dog_cat\train2

import numpy as np
import pandas as pd

# 이미지 데이터 전처리를 위한 클래스
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# 모델 구성에 필요한 클래스
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation

# 정확도 측정을 위한 함수
from sklearn.metrics import accuracy_score

# 🔸 데이터 전처리: 픽셀값을 0~1로 정규화
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 🔸 데이터가 저장된 경로 설정
path_train = './_data/kaggle/dog_cat/train2'
path_test = './_data/kaggle/dog_cat/test2'

# 🔸 훈련용 이미지 데이터 불러오기
xy_train = train_datagen.flow_from_directory(
    path_train,            # 훈련 데이터 폴더 경로
    target_size=(70, 70),  # 이미지 크기 통일
    batch_size=25000,      # 전체 데이터를 한 번에 가져오기
    class_mode='binary',   # 이진 분류 (강아지 vs 고양이)
    color_mode='rgb',      # RGB 컬러 이미지
    shuffle=True           # 데이터를 섞어서 가져오기
)

# 🔸 테스트용 이미지 데이터 불러오기
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(70, 70),
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
    # shuffle=True            
)

# 🔸 데이터 구조 확인
print(len(xy_train))    # 훈련 데이터 배치 수
print(len(xy_test))     # 테스트 데이터 배치 수
print(xy_train[0][0])   # 첫 번째 배치의 이미지 배열
print(xy_train[0][0].shape)  # (전체 개수, 70, 70, 3)
print(xy_train[0][1])   # 첫 번째 배치의 라벨 (0 또는 1)
print(xy_train[0][1].shape)

# 🔸 실제 데이터 변수에 저장
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

exit()

########################################################################
# 🔹 CNN 모델 구성
model = Sequential()

# 🔸 Block 1
model.add(Conv2D(32, (3,3), padding='same', input_shape=(70, 70, 3))) # 32개의 3x3 필터
model.add(BatchNormalization())  # 정규화
model.add(Activation('relu'))    # 비선형 활성화 함수
model.add(MaxPooling2D())        # 다운샘플링
model.add(Dropout(0.2))          # 과적합 방지

# 🔸 Block 2
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

# 🔸 Block 3
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

# 🔸 완전 연결층 (FC Layer)
model.add(Flatten())                    # 1차원으로 펼치기
model.add(Dense(256, activation='relu')) # 은닉층
model.add(Dropout(0.5))                 # 더 강한 드롭아웃

# 🔸 출력층
model.add(Dense(1, activation='sigmoid')) # 이진 분류이므로 sigmoid 사용

########################################################################
# 🔹 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 🔹 조기 종료 콜백
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)

# 🔹 모델 학습
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,  # 20%는 검증용으로 사용
    callbacks=[es],
    verbose=1
)

# 🔹 테스트 데이터로 평가
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("acc:", acc)

###############################################
# 🔹 예측값 생성 및 이진화
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # 확률 -> 0 or 1

# 🔹 성능 지표 출력
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔹 이미지 시각화
aaa = 1
import matplotlib.pyplot as plt
plt.imshow(x_train[aaa], 'gray')
plt.show()

###############################################
# 🔹 학습 과정 시각화 (Loss & Accuracy 그래프)

plt.figure(figsize=(12, 5))

# 🔸 Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 🔸 Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], label='Train Acc')
plt.plot(hist.history['val_acc'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#### 혼동 행렬 시각화
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')  # 예측 클래스
plt.ylabel('Actual')     # 실제 클래스
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred))

#### ROC Curve 시각화 (이진 분류에서 성능 확인)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # 기준선
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()
