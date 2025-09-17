import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

#1. 데이터
path = './_data/dacon/cancer/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# Label Encoding (Country, Race 포함)
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer'] 

# train_test_split 시 stratify=y 적용 (클래스 분포 유지)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=777,
    shuffle=True,
    stratify=y
)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 클래스 가중치 계산 (훈련 데이터 기준)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True,
)
hist = model.fit(
    x_train, y_train, 
    epochs=400, batch_size=128,
    validation_split=0.1,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

path_1 = './_save/keras23/'
model.save(path_1 + 'keras23_1_2save.h5')

#4. 평가 및 최적 threshold 찾기
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

y_proba = model.predict(x_test).ravel()

thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    score = f1_score(y_test, y_pred)
    f1_scores.append(score)

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"🔍 최적 Threshold: {best_thresh:.2f}, F1 Score: {best_f1:.4f}")

# 🔴 최적 threshold로 재평가
y_pred_optimal = (y_proba >= best_thresh).astype(int)

acc = accuracy_score(y_test, y_pred_optimal)
f1 = f1_score(y_test, y_pred_optimal)

print(f"📊 Accuracy (최적 threshold): {acc:.4f}")
print(f"📊 F1 Score (최적 threshold): {f1:.4f}")

# 🔴 F1 score vs threshold 시각화
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.axvline(x=best_thresh, color='r', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
plt.title("Threshold vs. F1 Score")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)


#5. 제출 파일 생성
y_submit_proba = model.predict(test_csv_scaled).ravel()
y_submit = (y_submit_proba >= best_thresh).astype(int)  # 🔴 최적 threshold 적용

submission_csv['Cancer'] = y_submit
submission_csv.to_csv('./_save/submit_0601_0900.csv')  # 🔴 제출 파일 저장
print("✅ 제출 파일 저장 완료: submit_0601_0800.csv")

"""
🔍 최적 Threshold: 0.77, F1 Score: 0.4860
📊 Accuracy (최적 threshold): 0.8835
📊 F1 Score (최적 threshold): 0.4860
1444/1444 [==============================] - 1s 453us/step
✅ 제출 파일 저장 완료: submit_best_thresh.csv


🔍 최적 Threshold: 0.55, F1 Score: 0.4891
📊 Accuracy (최적 threshold): 0.8842
📊 F1 Score (최적 threshold): 0.4891
1444/1444 [==============================] - 1s 470us/step
✅ 제출 파일 저장 완료: submit_0601_0833.csv
"""