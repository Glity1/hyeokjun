import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. 저장된 npy 데이터 불러오기
np_path = 'c:/study25/_data/_save_npy/'

x = np.load(np_path + "keras44_01_x_train.npy")
y = np.load(np_path + "keras44_01_y_train.npy")
print("x.shape:", x.shape, "| y.shape:", y.shape)

# 2. train / val 나누기
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train = x_train.reshape(x_train.shape[0], 100*100, 3)
x_test = x_val.reshape(x_val.shape[0], 100*100, 3)

# 3. 모델 구성
model = Sequential([
    LSTM(32, input_shape=(10000,3)),
    BatchNormalization(), Activation('relu'),
    Dropout(0.2),
    
    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 4. 학습
hist = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[es],
    verbose=1
)

# 5. 평가
loss, acc = model.evaluate(x_val, y_val)
print(f"val_loss: {loss:.4f}, val_acc: {acc:.4f}")

###################################################################################

# 6. 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.legend(), plt.title('Loss'), plt.grid()

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='val acc')
plt.legend(), plt.title('Accuracy'), plt.grid()
plt.tight_layout()
plt.show()

# ✅ 7. Confusion Matrix & ROC Curve
y_val_prob = model.predict(x_val)
y_val_pred = (y_val_prob > 0.5).astype(int)

cm = confusion_matrix(y_val, y_val_pred)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"), plt.ylabel("Actual")
plt.show()

print(classification_report(y_val, y_val_pred))

fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='AUC = %.2f' % roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('ROC Curve')
plt.legend(), plt.grid()
plt.show()

# ✅ 8. test2 이미지 예측
test_path = './_data/kaggle/dog_cat/test2/test/'
file_names = sorted(os.listdir(test_path))  # ex: 1.jpg, 2.jpg ...

x_pred = []
for fname in file_names:
    img = load_img(os.path.join(test_path, fname), target_size=(100, 100))
    img = img_to_array(img) / 255.0
    x_pred.append(img)

x_pred = np.array(x_pred)
print("x_pred.shape:", x_pred.shape)

y_pred_prob = model.predict(x_pred)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# ✅ 9. submission.csv 생성
submission = pd.DataFrame({
    'id': [int(fname.split('.')[0]) for fname in file_names],
    'label': y_pred
})
submission = submission.sort_values('id')  # 정렬 보정
submission.to_csv('./_data/kaggle/dog_cat/submission.csv', index=False)

print("✅ submission.csv 저장 완료!")
