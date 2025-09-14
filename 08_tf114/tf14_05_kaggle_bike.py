import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(777)

#1. 데이터
path = './_data/kaggle/bike/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0) # datetime column을 index 로 변경 아래 세줄 동일
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0) 

x = train_csv.drop(['count'], axis=1)
y = train_csv['count'].values


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813
)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# reshape target
y_train = y_train.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_train.shape[1],1]), name = 'weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

#2. 모델구성
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],feed_dict={x:x_train, y:y_train.reshape(-1,1)})
    if step % 500 == 0:
         print(step, '\t', cost_val)

# 결과 출력

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
sess.close()

# R2, MAE 계산
r2 = r2_score(y_test, y_predict)   # 실제값 vs 훈련데이터 예측값
mse = mean_squared_error(y_test, y_predict)

print('r2_score : ', r2) 
print('mse : ', mse)

# r2_score :  0.9999999995989625
# mse :  1.2934059884931116e-05