import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

path = './_data/dacon/cancer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# Label Encoding
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 스케일링 준비 (fit은 전체 데이터에 한번만)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# StratifiedKFold 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 앙상블 예측 저장
test_preds = np.zeros(len(test_csv))
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x_scaled, y)):
    print(f'Fold {fold+1}')
    x_train, x_val = x_scaled[train_idx], x_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # 모델 구성
    model = Sequential([
        Dense(64, input_dim=x_train.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=400, batch_size=128, callbacks=[es], class_weight=class_weight_dict, verbose=2)

    # val 예측 및 최적 threshold 찾기
    y_val_proba = model.predict(x_val).ravel()
    from sklearn.metrics import precision_recall_curve
    prec, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    val_f1 = f1_score(y_val, y_val_pred)
    print(f'Fold {fold+1} F1 Score: {val_f1:.4f} (threshold={best_threshold:.4f})')
    val_f1_scores.append(val_f1)

    # 테스트셋 예측 누적 (평균 예측 위해)
    test_pred_proba = model.predict(test_scaled).ravel()
    test_preds += test_pred_proba

# 평균 예측
test_preds /= skf.n_splits

# 전체 validation F1 score 평균
print(f'Average Validation F1 Score: {np.mean(val_f1_scores):.4f}')

# threshold는 보통 validation 평균 threshold로 설정하거나 0.5 사용
final_threshold = 0.5

submission_csv['Cancer'] = (test_preds >= final_threshold).astype(int)
submission_csv.to_csv(path + 'submission_kfold_ensemble2.csv')
