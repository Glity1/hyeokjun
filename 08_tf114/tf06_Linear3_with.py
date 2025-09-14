import tensorflow as tf
# sess = tf.compat.v1.Session()

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [4, 6, 8,10,12]

w = tf.compat.v1.Variable(111, dtype=tf.float32)
b = tf.compat.v1.Variable(0.0, dtype=tf.float32)
# Tensor1 에서는 모델 돌려면 w, b까지 준비해줘야한다 

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
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))

#4. 평가 예측
