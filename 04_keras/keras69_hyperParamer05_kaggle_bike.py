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
path = './_data/kaggle/bike/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0) # datetime column을 index 로 변경 아래 세줄 동일
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
samplesubmission_csv = pd.read_csv(path + 'samplesubmission.csv')

x = train_csv.drop(['count', 'casual', 'registered'], axis=1) 

y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74, # validation_split로도 바뀌지않는다면 바꾸자
    )

#2. 모델구성
def build_model(drop=0.5, optimizer='adam',activation='relu',
                 node1=128, node2=64, node3=32, node4=16, node5=9, lr=0.001):
    inputs = Input(shape=(8,), name='inputs')
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

# - 최적의 매개변수 :  <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000246477E3F10>
# - 최적의 파라미터 :  {'optimizer': 'rmsprop', 'node5': 8, 'node4': 32, 'node3': 16, 'node2': 16, 'node1': 64, 'lr': 0.0001, 'drop': 0.2, 'batch_size': 32, 'activation': 'linear'}
# - best_score :  -24894.072265625
# 69/69 [==============================] - 0s 418us/step - loss: 25046.5703 - mse: 25046.5703
# - mode.score :  -25046.5703125
# 69/69 [==============================] - 0s 236us/step
# - r2 :  0.22385108458616898
# 69/69 [==============================] - 0s 402us/step
# - best_r2_score :  0.22385108458616898
# - 걸린시간 :  19.47 초