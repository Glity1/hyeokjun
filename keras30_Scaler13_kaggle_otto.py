# https://www.kaggle.com/c/otto-group-product-classification-challenge/overview

import numpy as np                                           
import pandas as pd                                          
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(sampleSubmission_csv)             # [61878 rows x 94 columns]

print(train_csv)                        # [144368 rows x 93 columns]
print(train_csv.columns)                # Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
                                        #    'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13',
                                        #    'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19',
                                        #    'feat_20', 'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25',
                                        #    'feat_26', 'feat_27', 'feat_28', 'feat_29', 'feat_30', 'feat_31',
                                        #    'feat_32', 'feat_33', 'feat_34', 'feat_35', 'feat_36', 'feat_37',
                                        #    'feat_38', 'feat_39', 'feat_40', 'feat_41', 'feat_42', 'feat_43',
                                        #    'feat_44', 'feat_45', 'feat_46', 'feat_47', 'feat_48', 'feat_49',
                                        #    'feat_50', 'feat_51', 'feat_52', 'feat_53', 'feat_54', 'feat_55',
                                        #    'feat_56', 'feat_57', 'feat_58', 'feat_59', 'feat_60', 'feat_61',
                                        #    'feat_62', 'feat_63', 'feat_64', 'feat_65', 'feat_66', 'feat_67',
                                        #    'feat_68', 'feat_69', 'feat_70', 'feat_71', 'feat_72', 'feat_73',
                                        #    'feat_74', 'feat_75', 'feat_76', 'feat_77', 'feat_78', 'feat_79',
                                        #    'feat_80', 'feat_81', 'feat_82', 'feat_83', 'feat_84', 'feat_85',
                                        #    'feat_86', 'feat_87', 'feat_88', 'feat_89', 'feat_90', 'feat_91',
                                        #    'feat_92', 'feat_93'],
                                        #   dtype='object') 
                                                             
print(test_csv)                         # [144368 rows x 10 columns]
print(test_csv.columns)                 # Index(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
                                        #    'Class_7', 'Class_8', 'Class_9'],
                                        #   dtype='object')
print(sampleSubmission_csv)             # <class 'pandas.core.frame.DataFrame'>
print(sampleSubmission_csv.columns)     # Int64Index: 61878 entries, 1 to 61878 
print(sampleSubmission_csv.head())


# 결측치 확인
    
print(train_csv.info())                 
print(train_csv.isnull().sum())         #결측치 없음
print(test_csv.isna().sum())            #결측치 없음

print(train_csv.describe())             #[8 rows x 93 columns]


# #################### x,y 데이터를 분리한다 ##########################
print('###############################################################################')
x = train_csv.drop(['target'], axis=1) # target column만 빼겠다.
print(x)                               # [61878 rows x 93 columns]
                       
y = train_csv['target']                # target column만 빼서 y에 넣겠다.
print(y)  
print(y.shape)                         # (61878,)

le = LabelEncoder()
y = le.fit_transform(y)
# exit()

x_train, x_val, y_train, y_val = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74,
    stratify=y
    )

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
test_csv_scaled = scaler.transform(test_csv)

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(x_train, y_train)

importances = pd.Series(lgb_model.feature_importances_, index=x.columns)
importances = importances.sort_values(ascending=False)
all_features = x.columns.tolist()

x_train_final = pd.DataFrame(x_train_scaled, columns=x.columns)[all_features].values  
x_val_final = pd.DataFrame(x_val_scaled, columns=x.columns)[all_features].values      
test_final = pd.DataFrame(test_csv_scaled, columns=test_csv.columns)[all_features].values

# #2. 모델구성

model = Sequential()
model.add(Dense(512, input_shape=(93,), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.05))  
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))  
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))  
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(9, activation='softmax'))

path='./_save/keras30/Scaler13_kaggle_otto/'
model.save_weights(path+'keras30_Scaler13_kaggle_otto_weights.h5')

#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

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
    patience=30,                       
    restore_best_weights=True,        
) 

path='./_save/keras30/Scaler13_kaggle_otto/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras30_Scaler13_kaggle_otto.hdf5'
    )

hist = model.fit(x_train_final, y_train, epochs=300, batch_size=128,
             verbose=2, # 각 에포크마다 진행 상황 출력
             validation_data=(x_val_final, y_val), # 명시적으로 검증 데이터 사용
             callbacks=[es, mcp],
             class_weight=class_weights_dict # 클래스 가중치 적용
          )

path='./_save/keras30/Scaler13_kaggle_otto/'
model.save(path+'keras30_Scaler13_kaggle_otto.h5')

# 검증 데이터 예측 및 f1 score 계산
val_pred_probs = model.predict(x_val_final) 
val_preds = np.argmax(val_pred_probs, axis=1) # 확률 중 가장 높은 클래스 선택
val_f1 = f1_score(y_val, val_preds, average='macro') # Macro F1 Score 계산
print('\nValidation Macro F1 Score:', val_f1)

# 제출 파일 생성 부분 (수정됨)
test_pred_probs = model.predict(test_final)    

submission = sampleSubmission_csv.copy()
for i in range(9):
    col_name = f'Class_{i+1}'
    submission[col_name] = test_pred_probs[:, i] 

path='./_save/keras30/Scaler13_kaggle_otto/'
sampleSubmission_csv.to_csv(path + 'sampleSubmission_0608_2350.csv', index=False) # csv 만들기.
print("\n📁 'sampleSubmission_csv' 저장 완료!")

"""
#4. 평가, 예측
y_val_pred_probs = model.predict(x_val_top)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

# f1-score (macro)
f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation F1 Score (macro): {f1:.4f}")

# 14. 테스트 데이터 예측
test_pred_probs = model.predict(test_top)
test_pred = np.argmax(test_pred_probs, axis=1)

# 15. 예측 결과를 원래 클래스 이름으로 디코딩
test_pred_labels = le.inverse_transform(test_pred)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
sampleSubmission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit
#print(samplesubmission_csv)
# print(submission_csv)

######################## csv파일 만들기 ######################

"""
