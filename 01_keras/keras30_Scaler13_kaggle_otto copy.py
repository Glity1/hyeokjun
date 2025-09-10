import numpy as np                                        
import pandas as pd    
import tensorflow as tf                                   
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight 
import matplotlib.pyplot as plt
import os 

#1. 데이터 로드
path = './_data/kaggle/otto/'   

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print("train_csv shape:", train_csv.shape)
print("test_csv shape:", test_csv.shape) 
print("sampleSubmission_csv shape:", sampleSubmission_csv.shape)

print("\n--- train_csv head ---")
print(train_csv.head())
print("\n--- test_csv head (feat_X 컬럼이 보여야 함) ---")
print(test_csv.head()) 
print("\n--- sampleSubmission_csv head ---")
print(sampleSubmission_csv.head())


# #################### x,y 데이터 분리 ##########################
x = train_csv.drop(['target'], axis=1) 
y = train_csv['target'] 
                                
# 타겟 변수 Label Encoding
le = LabelEncoder()
y = le.fit_transform(y) 

# 훈련 및 검증 데이터 분리 (validation_data 사용을 위해 명시적으로 분리)
x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.2, 
    random_state=74,
    stratify=y 
)

# test_csv에서 'id' 컬럼 분리 (이제 인덱스에서 가져옵니다)
test_ids = test_csv.index # test_csv의 인덱스를 그대로 사용
test_features = test_csv # 'id'가 이미 인덱스이므로, test_csv 자체가 특성 데이터입니다.

# 데이터 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_val_scaled = scaler.transform(x_val)         
test_features_scaled = scaler.transform(test_features)   

# 최종 데이터셋 구성 (NumPy 배열) - 모든 특성 사용
x_train_final = x_train_scaled  # 이미 스케일링된 x_train은 모든 특성을 포함
x_val_final = x_val_scaled      # 이미 스케일링된 x_val은 모든 특성을 포함
test_final = test_features_scaled # 이미 스케일링된 test_features는 모든 특성을 포함

# #################### 2. 모델 구성 ##########################
# 입력 특성 수가 93개이므로 input_shape를 (93,)으로 설정
model = Sequential()
model.add(Dense(512, input_shape=(93,), activation='relu'))
model.add(BatchNormalization()) # Dense layer 후에 BatchNormalization
model.add(Dropout(0.05))   

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.05))   

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.05))   

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.1))   

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.1))   

model.add(Dense(9, activation='softmax')) # 9개의 클래스 (0~8)

save_path='./_save/keras30/otto_all_features_no_lgbm/' # 저장 경로 변경
os.makedirs(save_path, exist_ok=True) # 폴더 없으면 생성

model.save_weights(save_path+'otto_all_features_no_lgbm_weights.h5')

# #################### 3. 컴파일, 훈련 ##########################
# 학습률을 더 낮게 설정하여 안정적인 학습 유도
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

# Class Weight 계산
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print("\nComputed Class Weights:", class_weights_dict)

es = EarlyStopping(          
    monitor='val_loss',
    mode = 'min',            
    patience=50, # patience를 50으로 더 증가
    restore_best_weights=True,    
) 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=save_path+'otto_all_features_no_lgbm_best.hdf5'
)

print("\n--- 모델 학습 시작 ---")
hist = model.fit(x_train_final, y_train, epochs=500, batch_size=128, # epochs를 500으로 증가
             verbose=2, 
             validation_data=(x_val_final, y_val),
             callbacks=[es, mcp],
             class_weight=class_weights_dict 
)
print("--- 모델 학습 완료 ---")

model.save(save_path+'otto_all_features_no_lgbm_final_model.h5')

# #################### 4. 평가, 예측 및 제출 파일 생성 ##########################

# 검증 데이터 예측 및 f1 score 계산
val_pred_probs = model.predict(x_val_final) 
val_preds = np.argmax(val_pred_probs, axis=1) 
val_f1 = f1_score(y_val, val_preds, average='macro') 
print('\nValidation Macro F1 Score:', val_f1)

# 테스트 데이터 예측
test_pred_probs = model.predict(test_final)   

# 제출 파일 생성
submission = sampleSubmission_csv.copy() 
submission['id'] = test_ids # 저장해둔 test_ids 다시 할당

for i in range(9):
    col_name = f'Class_{i+1}'
    submission[col_name] = test_pred_probs[:, i] 

submission.to_csv(save_path + 'sampleSubmission_all_features_no_lgbm.csv', index=False) 
print(f"\n📁 '{save_path}sampleSubmission_all_features_no_lgbm.csv' 저장 완료!")

