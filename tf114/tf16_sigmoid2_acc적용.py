import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(777)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

#2. 모델구성
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1-y) * tf.math.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.8)
train = optimizer.minimize(loss)

pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# acc 만들기
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
epochs = 3501
for step in range(epochs) : 
    cost_val, _, w_val, b_val, accuracy = sess.run([loss, train, w, b, acc], feed_dict={x:x_data, y:y_data})
    if step % 50 ==0:
        print(step, 'loss : ', cost_val, 'acc : ', accuracy)

print('최종 weight : ', w_val, 'bias : ', b_val)
print('acc : ', acc)
