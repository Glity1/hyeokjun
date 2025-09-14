import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras. optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from sklearn. preprocessing import MinMaxScaler, StandardScaler
from sklearn. metrics import accuracy_score, r2_score
from sklearn. datasets import load_diabetes
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
dataset_csv.index = pd.to_datetime(dataset_csv.index)

# 입력 데이터 x, 타겟 데이터 y 생성
x = dataset_csv[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y = dataset_csv['wd (deg)']


# MinMaxScaler 적용
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 2. 시퀀스 데이터 생성 함수 정의
def split_xy(x, y, timesteps, target_steps, stride):
    x_seq, y_seq = [], []
    for i in range(0, len(x) - timesteps - target_steps + 1, stride):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + target_steps])
    return np.array(x_seq), np.array(y_seq)

# 3. 시퀀스 구성 
timesteps = 144
target_steps = 144
stride = 1
x_seq, y_seq = split_xy(x_scaled, y.to_numpy(), timesteps, target_steps, stride)

# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=222, shuffle=True
)
#2. 모델구성
def build_model(drop=0.5, optimizer='adam',activation='relu',
                 node1=128, node2=64, node3=32, node4=16, node5=9, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    output = Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=output )

    model.compile(optimizer=optimizer, metrics=['mse'], loss='mse')
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    lr = [0.1, 0.01, 0.001, 0.0001]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear' ]
    node1 = [128, 64, 32, 16] 
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16] 
    node4 = [128, 64, 32, 16] 
    node5 = [128, 64, 32, 16, 8]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'lr' : lr,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,

    }

hyperparameters = create_hyperparameter()
print(hyperparameters)



from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3,
                           n_iter=2,
                           verbose=1,
                           )

es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=12,
                   restore_best_weights=True)

rl = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, epochs=20, validation_split= 0.1, verbose=1, callbacks = [es, rl])
end = time.time() - start

# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)            # refit=false 상태일 때 print다 안됨
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))
 
y_pred = model.predict(x_test)                                      # 두 predict 두개중에 원하는거 쓰면된다
print('- r2 : ', r2_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('- best_r2_score : ', r2_score(y_test, y_pred_best))


print('- 걸린시간 : ', round(end, 2), '초\n\n')

# - 최적의 매개변수 :  <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000217B45EF280>
# - 최적의 파라미터 :  {'optimizer': 'rmsprop', 'node5': 128, 'node4': 64, 'node3': 64, 
# 'node2': 16, 'node1': 16, 'drop': 0.2, 'batch_size': 16, 'activation': 'linear'}

# - best_r2_score :  0.41194842133562093
# - 걸린시간 :  6.31 초