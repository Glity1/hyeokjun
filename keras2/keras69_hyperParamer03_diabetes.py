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
x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=337
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

# 리펙토링 버전
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import time

# 1. 데이터 로드
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337)

# 2. 모델 구성 함수
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,))
    x = Dense(node1, activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation)(x)
    x = Dense(node5, activation=activation)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)

    # 옵티마이저 매핑
    optimizer_dict = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'adadelta': Adadelta(learning_rate=lr)
    }

    model.compile(optimizer=optimizer_dict[optimizer], loss='mse', metrics=['mse'])
    return model

# 3. 하이퍼파라미터 공간 정의
def create_hyperparameter():
    return {
        'batch_size': [32, 16, 8, 1, 64],
        'optimizer': ['adam', 'rmsprop', 'adadelta'],
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'drop': [0.2, 0.3, 0.4, 0.5],
        'activation': ['relu', 'elu', 'selu', 'linear'],
        'node1': [128, 64, 32, 16],
        'node2': [128, 64, 32, 16],
        'node3': [128, 64, 32, 16],
        'node4': [128, 64, 32, 16],
        'node5': [128, 64, 32, 16, 8],
    }

hyperparameters = create_hyperparameter()

# 4. 콜백 정의
es = EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1)

# 5. 래핑 및 서치
keras_model = KerasRegressor(build_fn=build_model, verbose=0)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=2, verbose=1)

# 6. 학습
start = time.time()
model.fit(x_train, y_train,
          epochs=20,
          validation_split=0.1,
          callbacks=[es, rl],
          verbose=1)
end = time.time() - start

# 7. 결과 출력
print("\n✅ 최적의 파라미터:", model.best_params_)
print("✅ Best CV Score (train r2):", model.best_score_)
print("✅ Test Score (r2):", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("✅ Test R2:", r2_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print("✅ Best Estimator R2:", r2_score(y_test, y_pred_best))
print("⏱️ 학습 소요 시간:", round(end, 2), "초")