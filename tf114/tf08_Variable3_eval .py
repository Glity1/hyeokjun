# tf07.copy
# 실습 : 변수 초기화를 interactivesession .eval 로 변경

import tensorflow as tf

tf.random.set_random_seed(111)
# sess = tf.compat.v1.Session()

# 1. 데이터
x_data = [1, 2, 3, 4, 5]
y_data = [4, 6, 8,10,12]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

x_test_data = [6, 7, 8]
x_test = tf.placeholder(tf.float32, shape=[None])

# weight 및 bias 초기값을 지정해서 줌
# w = tf.compat.v1.Variable(111, dtype=tf.float32)
# b = tf.compat.v1.Variable(0.0, dtype=tf.float32)
# Tensor1 에서는 모델 돌려면 w, b까지 준비해줘야한다

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# print(w)
# exit()

#2. 모델 (y = wx + b)
hypothesis = x * w  + b

#3. 컴파일 훈련
# model.compile(loss = 'mse', optimizer = 'sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

# model.fit()
# with문 내부에서만 sess 실행
# sess.close() 생략 가능
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 4000
    
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict = {x:x_data,
                                                          y:y_data})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)

    #4. 평가 예측 [1] : Placeholder
    pred = x_test * w_val + b_val

    print("결과", sess.run(pred, feed_dict = {x_test : x_test_data}))
    # [14.00000942 16.00001299 18.00001657]
    
#4. 평가 예측 [2] : Python, Numpy
pred_2 = x_test_data * w_val + b_val
print(pred_2)
