import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random
import os

# 저장 폴더 생성
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

# 1. 데이터 불러오기
path = './_data/dacon/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 전처리
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])

X = train.drop(['Cancer'], axis=1)
y = train['Cancer']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# 3. Seed variation
seeds = [42, 777, 2024, 3407, 999, 1313, 222, 333, 444, 555, 888, 2025, 31415]

for seed in seeds:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_train, x_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=seed
    )

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=400, batch_size=128,
                     validation_split=0.1, callbacks=[es], 
                     class_weight=class_weight_dict, verbose=0)

    y_val_pred_prob = model.predict(x_val).ravel()
    prec, recall, thresholds = precision_recall_curve(y_val, y_val_pred_prob)
    f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    y_val_pred = (y_val_pred_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)

    print(f"SEED: {seed}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Validation F1 Score: {best_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 50)

    # 모델 및 가중치 저장
    model_save_path = os.path.join(save_dir, f'model_seed_{seed}.h5')
    model.save(model_save_path)
