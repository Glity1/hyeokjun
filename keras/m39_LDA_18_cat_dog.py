import numpy as np
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 저장된 npy 데이터 불러오기
np_path = './_data/_save_npy/'
x = np.load(np_path + "keras44_01_x_train.npy")    # shape 예: (20000, 150, 150, 3)
y = np.load(np_path + "keras44_01_y_train.npy")

# 2. Pillow로 이미지 resize (64x64)
target_size = (64, 64)
x_resized = np.array([
    np.array(Image.fromarray(img.astype(np.uint8)).resize(target_size, Image.BILINEAR))
    for img in x
])  # shape: (20000, 64, 64, 3)

# 3. train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_resized, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 5. Flatten
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 6. One-hot encode labels
y_train_cat = pd.get_dummies(y_train.reshape(-1))
y_test_cat = pd.get_dummies(y_test.reshape(-1))

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# 7. 클래스 수 확인
n_classes = len(np.unique(y_train))

# 8. LDA 실험 반복
nums = [1, 30, 60, 99]
for i in nums:
    n_components = min(i, n_classes - 1)  # LDA 최대 차원 제한
    print(f"\n====== lda n_components = {n_components} (요청: {i}) ======")

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

    # 9. 모델 구성
    model = Sequential()
    model.add(Dense(128, input_dim=n_components, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))

    # 10. 컴파일 및 학습
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

    start = time.time()
    model.fit(x_train_lda, y_train_cat, epochs=300, batch_size=128,
              validation_split=0.2, callbacks=[es], verbose=1)
    end = time.time()

    # 11. 평가 및 예측
    loss, acc = model.evaluate(x_test_lda, y_test_cat, verbose=1)
    y_pred = model.predict(x_test_lda, verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)
    acc_score = accuracy_score(y_test, y_pred_label)

