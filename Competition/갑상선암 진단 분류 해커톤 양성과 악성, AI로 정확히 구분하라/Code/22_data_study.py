import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
path = './_data/dacon/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 전처리
binary_cols = ['Gender','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']
le = LabelEncoder()
for col in binary_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

multi_cols = ['Country', 'Race']
train = pd.get_dummies(train, columns=multi_cols)
test = pd.get_dummies(test, columns=multi_cols)
for col in set(train.columns)-set(test.columns):
    test[col] = 0
test = test[train.drop('Cancer', axis=1).columns]

X = train.drop('Cancer', axis=1)
y = train['Cancer']

# 파생변수
eps = 1e-8
X['T4_TSH_Ratio'] = X['T4_Result'] / (X['TSH_Result'] + eps)
X['T3_T4_Ratio']  = X['T3_Result'] / (X['T4_Result'] + eps)
X['T3_TSH_Ratio'] = X['T3_Result'] / (X['TSH_Result'] + eps)
X['Age_Nodule_Interaction'] = X['Age'] * X['Nodule_Size']

test['T4_TSH_Ratio'] = test['T4_Result'] / (test['TSH_Result'] + eps)
test['T3_T4_Ratio']  = test['T3_Result'] / (test['T4_Result'] + eps)
test['T3_TSH_Ratio'] = test['T3_Result'] / (test['TSH_Result'] + eps)
test['Age_Nodule_Interaction'] = test['Age'] * test['Nodule_Size']

# 스케일링
scale_cols = ['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result','T4_TSH_Ratio','T3_T4_Ratio','T3_TSH_Ratio','Age_Nodule_Interaction']
scaler = MinMaxScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
test[scale_cols] = scaler.transform(test[scale_cols])

# train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# class_weight 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Focal Loss
def focal_loss(gamma=1.5, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt + K.epsilon()))
    return loss

# 모델
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=focal_loss(), optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(patience=20, restore_best_weights=True)

model.fit(X_train, y_train, epochs=300, batch_size=256, validation_data=(X_val, y_val),
          class_weight=class_weight_dict, callbacks=[es], verbose=2)

# Threshold 튜닝
y_val_pred = model.predict(X_val).ravel()
prec, recall, thresholds = precision_recall_curve(y_val, y_val_pred)
f1s = 2*(prec*recall)/(prec+recall+1e-8)
best_th = thresholds[np.argmax(f1s)]
print(f"Best threshold: {best_th:.4f}")
print(f"Validation F1 Score: {np.max(f1s):.4f}")

# 의사라벨링
test_pred_proba = model.predict(test).ravel()
pseudo_idx = np.where((test_pred_proba > 0.8) | (test_pred_proba < 0.2))[0]
pseudo_labels = (test_pred_proba[pseudo_idx] > 0.5).astype(int)

X_pseudo = test.iloc[pseudo_idx]
y_pseudo = pseudo_labels

# Pseudo-label 포함 재학습
X_train_final = pd.concat([X_train, X_pseudo])
y_train_final = pd.concat([y_train, pd.Series(y_pseudo)])

model.fit(X_train_final, y_train_final, epochs=300, batch_size=256,
          class_weight=class_weight_dict, callbacks=[es], verbose=2)

# 제출파일
final_test_pred = model.predict(test).ravel()
final_test_labels = (final_test_pred >= best_th).astype(int)
sub['Cancer'] = final_test_labels
sub.to_csv(path + 'final_submission.csv')

# 저장
model.save('./_save/final_thyroid_model.h5')

# 시각화
plt.plot(prec, recall)
plt.title("Precision-Recall Curve")
plt.show()

plt.plot(thresholds, f1s[:-1])
plt.title("Threshold vs F1")
plt.show()

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.show()

