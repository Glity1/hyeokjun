import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
path = './_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 라벨인코딩
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

# 파생변수 생성
epsilon = 1e-6
train_csv['T4_TSH_Ratio'] = train_csv['T4_Result'] / (train_csv['TSH_Result'] + epsilon)
train_csv['T3_T4_Ratio'] = train_csv['T3_Result'] / (train_csv['T4_Result'] + epsilon)
train_csv['Age_Nodule_Interaction'] = train_csv['Age'] * train_csv['Nodule_Size']

test_csv['T4_TSH_Ratio'] = test_csv['T4_Result'] / (test_csv['TSH_Result'] + epsilon)
test_csv['T3_T4_Ratio'] = test_csv['T3_Result'] / (test_csv['T4_Result'] + epsilon)
test_csv['Age_Nodule_Interaction'] = test_csv['Age'] * test_csv['Nodule_Size']

# feature, target 분리
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# train/test 분리
x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
test_csv_scaled = scaler.transform(test_csv)

# focal loss 정의
def focal_loss(alpha=0.4, gamma=2.0):
    def focal(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * K.pow((1 - p_t), gamma) * bce
        return K.mean(loss)
    return focal

# 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# 컴파일
model.compile(loss=focal_loss(alpha=0.4, gamma=2.0), optimizer=Adam(learning_rate=0.001))

# 콜백
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# 학습
hist = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=300,
    batch_size=128,
    callbacks=[es],
    verbose=2
)

# Threshold 튜닝
y_pred_prob = model.predict(x_valid).ravel()
precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"최적 threshold: {best_threshold:.4f}")
print(f"Validation F1 Score: {f1_scores[best_idx]:.4f}")

# Confusion matrix
y_pred_final = (y_pred_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_valid, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 제출
test_pred_prob = model.predict(test_csv_scaled).ravel()
test_pred = (test_pred_prob >= best_threshold).astype(int)
submission_csv['Cancer'] = test_pred
submission_csv.to_csv(path + 'submission_focal.csv')
