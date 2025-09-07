# https://www.kaggle.com/c/otto-group-product-classification-challenge/overview

import numpy as np                                           
import pandas as pd                                          
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
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

# print(sampleSubmission_csv)             # [61878 rows x 94 columns]

# print(train_csv)                        # [144368 rows x 93 columns]
# print(train_csv.columns)                # Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
#                                         #    'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13',
#                                         #    'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19',
#                                         #    'feat_20', 'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25',
#                                         #    'feat_26', 'feat_27', 'feat_28', 'feat_29', 'feat_30', 'feat_31',
#                                         #    'feat_32', 'feat_33', 'feat_34', 'feat_35', 'feat_36', 'feat_37',
#                                         #    'feat_38', 'feat_39', 'feat_40', 'feat_41', 'feat_42', 'feat_43',
#                                         #    'feat_44', 'feat_45', 'feat_46', 'feat_47', 'feat_48', 'feat_49',
#                                         #    'feat_50', 'feat_51', 'feat_52', 'feat_53', 'feat_54', 'feat_55',
#                                         #    'feat_56', 'feat_57', 'feat_58', 'feat_59', 'feat_60', 'feat_61',
#                                         #    'feat_62', 'feat_63', 'feat_64', 'feat_65', 'feat_66', 'feat_67',
#                                         #    'feat_68', 'feat_69', 'feat_70', 'feat_71', 'feat_72', 'feat_73',
#                                         #    'feat_74', 'feat_75', 'feat_76', 'feat_77', 'feat_78', 'feat_79',
#                                         #    'feat_80', 'feat_81', 'feat_82', 'feat_83', 'feat_84', 'feat_85',
#                                         #    'feat_86', 'feat_87', 'feat_88', 'feat_89', 'feat_90', 'feat_91',
#                                         #    'feat_92', 'feat_93'],
#                                         #   dtype='object') 
                                                             
# print(test_csv)                         # [144368 rows x 10 columns]
# print(test_csv.columns)                 # Index(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
#                                         #    'Class_7', 'Class_8', 'Class_9'],
#                                         #   dtype='object')
# print(sampleSubmission_csv)             # <class 'pandas.core.frame.DataFrame'>
# print(sampleSubmission_csv.columns)     # Int64Index: 61878 entries, 1 to 61878 
# print(sampleSubmission_csv.head())


# 결측치 확인
    
# print(train_csv.info())                 
# print(train_csv.isnull().sum())         #결측치 없음
# print(test_csv.isna().sum())            #결측치 없음

# print(train_csv.describe())             #[8 rows x 93 columns]


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

print(x_train.shape, x_val.shape)  # (49502, 93) (12376, 93)
print(y_train.shape, y_val.shape)  # (49502,) (12376,)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_val) 

# exit()

x_train = x_train.reshape(49502,31,3,1)
x_test = x_test.reshape(12376,31,3,1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(31, 3, 1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=9, activation='linear'))

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

hist = model.fit(x_train, y_train, epochs=300, batch_size=256,
             verbose=1, # 각 에포크마다 진행 상황 출력
             validation_data=(x_test, y_val), # 명시적으로 검증 데이터 사용
             callbacks=[es],
             class_weight=class_weights_dict # 클래스 가중치 적용
          )

# 검증 데이터 예측 및 f1 score 계산
val_pred_probs = model.predict(x_test) 
val_preds = np.argmax(val_pred_probs, axis=1) # 확률 중 가장 높은 클래스 선택
val_f1 = f1_score(y_val, val_preds, average='macro') # Macro F1 Score 계산
print('\nValidation Macro F1 Score:', val_f1)
