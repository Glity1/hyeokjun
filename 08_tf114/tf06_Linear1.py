import tensorflow as tf
sess = tf.compat.v1.Session()

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

w = tf.compat.v1.Variable(111, dtype=tf.float32)
b = tf.compat.v1.Variable(0.0, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
# Tensor1 에서는 모델 돌려면 w, b까지 준비해줘야한다 

#2. 모델 (y = wx + b)
hypothesis = x * w  + b

#3. 컴파일 훈련
# model.compile(loss = 'mse', optimizer = 'sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.15)
train = optimizer.minimize(loss)

# model.fit()
epochs = 4000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))

#4. 평가 예측

sess.close() # 메모리 누수 방지를 위해서 마지막에는 닫아준다.