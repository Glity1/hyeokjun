import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, y_train.shape) # (404, 13) (404,)
print(x_test.shape, y_test.shape)   # (102, 13) (102,)


# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# reshape target
y_train = y_train.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,1]), name = 'weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

#2. 모델구성
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3000
for step in range(epochs):
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b],
                                     feed_dict={x:x_train, y:y_train.reshape(-1,1)})
    if step % 500 == 0:
         print(step, '\t', cost_val)

# 결과 출력

#1. 기존방법
# y_predict = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_v) + b_v
# y_predict = sess.run(hypothesis, feed_dict={x:x_test})

#2. 재현방법
y_predict = np.matmul(x_test, w_v) + b_v

# R2, MAE 계산
r2 = r2_score(y_test, y_predict)   # 실제값 vs 훈련데이터 예측값
mse = mean_squared_error(y_test, y_predict)

print('r2_score : ', r2) 
print('mse : ', mse)

# r2_score :  0.6576401568066772
# mse :  28.499350926047473