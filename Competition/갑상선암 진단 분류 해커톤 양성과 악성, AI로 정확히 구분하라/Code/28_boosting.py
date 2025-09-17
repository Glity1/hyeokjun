# ================== 라이브러리 ==================
import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# 부스팅 모델
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# ================== 1. 데이터 로딩 및 전처리 ==================
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=777, shuffle=True, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ================== 2. 딥러닝 모델 ==================
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

path_1 = './_save/keras23/'
model.save_weights(path_1 + 'keras23_new_save_weights.h5')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=40, restore_best_weights=True)
model.fit(x_train, y_train, 
          epochs=400, batch_size=64, validation_split=0.1, 
          callbacks=[es], class_weight=class_weight_dict, verbose=2)


model.save(path_1 + 'keras23_new_save.h5')


# 딥러닝 예측 확률 및 threshold 조정
y_proba_dl = model.predict(x_test).ravel()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_dl)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh_dl = thresholds[best_idx]
y_pred_dl = (y_proba_dl >= best_thresh_dl).astype(int)

print(f"DL Best Threshold: {best_thresh_dl}")
print(f"DL Accuracy: {accuracy_score(y_test, y_pred_dl)}")
print(f"DL F1 Score: {f1_score(y_test, y_pred_dl)}")

"""
# ================== 3. 부스팅 모델 ==================
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=class_weight_dict[0] / class_weight_dict[1]
)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGB F1 Score:", f1_score(y_test, y_pred_xgb))

lgb_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    class_weight='balanced', random_state=42
)
lgb_model.fit(x_train, y_train)
y_pred_lgb = lgb_model.predict(x_test)
print("LGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LGBM F1 Score:", f1_score(y_test, y_pred_lgb))

cat_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6, verbose=0,
    random_state=42, class_weights=[class_weight_dict[0], class_weight_dict[1]]
)
cat_model.fit(x_train, y_train)
y_pred_cat = cat_model.predict(x_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost F1 Score:", f1_score(y_test, y_pred_cat))

# ================== 4. 앙상블 (VotingClassifier) ==================
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb_model),
        ('cat', cat_model),
    ],
    voting='soft'
)
voting.fit(x_train, y_train)
y_pred_vote = voting.predict(x_test)
print("Voting Accuracy:", accuracy_score(y_test, y_pred_vote))
print("Voting F1 Score:", f1_score(y_test, y_pred_vote))

# ================== 5. 제출 파일 생성 (Voting 기반) ==================
test_pred = voting.predict(test_csv_scaled)
submission_csv['Cancer'] = test_pred
path_2 = './_data/dacon/cancer/'
submission_csv.to_csv(path_2 + 'submission_1808.csv')
print("✅ submission_1808.csv 파일 생성 완료!")

# Voting Accuracy: 0.8838916934373566
# Voting F1 Score: 0.4849872773536896

"""
