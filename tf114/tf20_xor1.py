import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1-hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
epochs = 101
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], 
                                         feed_dict={x:x_data, y:y_data})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val,)

#4. 평가, 예측
y_predict = tf.sigmoid(tf.matmul(tf.cast(x_data, tf.float32), w_val) + b_val)
y_pred = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_data)
print('acc : ', acc)
