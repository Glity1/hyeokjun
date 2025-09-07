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
########################### optimizer #################################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

# loss = (hypothesis - y)**2 = (x * w - y)**2
# loss 를 w로 미분 = 2*(x * w - y ) * x

lr = 0.01
gradient = tf.reduce_mean((x * w + b - y) * x)
descent =  w - lr * gradient
train = w.assign(descent)    #assign : 값을 대체하겠다.

#3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

    print("===========================predict================================")
    y_predict = x_test * w_val + b_val
    print("[6,7,8]결과: ", sess.run(y_predict, feed_dict={x_test:x_test_data}))

plt.plot(loss_val_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()