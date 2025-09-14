#11-1copy

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(777)

import matplotlib.pyplot as plt

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random.normal([1], dtype=tf.float32))
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])


#2. 모델구성
hypothesis = x* w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 2001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

    print("===========================predict================================")
    y_predict = x_test * w_val + b_val
    print("[6,7,8]결과: ", sess.run(y_predict, feed_dict={x_test:x_test_data}))


################# 실습 : R2, mae 만들기 #################
from sklearn.metrics import r2_score, mean_absolute_error

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     epochs = 2001
#     for step in range(epochs):
#         sess.run(train, feed_dict={x: x_data, y: y_data})

#     # 최종 예측 (훈련 데이터 기준)
#     y_pred_train = sess.run(hypothesis, feed_dict={x: x_data})

#     # 테스트 예측
#     y_pred_test = sess.run(hypothesis, feed_dict={x: x_test_data})

y_predict = x_test_data * w_val + b_val
y_test = [13, 15, 17]

# R2, MAE 계산
r2 = r2_score(y_test, y_predict)   # 실제값 vs 훈련데이터 예측값
mae = mean_absolute_error(y_test, y_predict)

print('r2_score : ', r2) 
print('mae : ', mae)