#20-1 copy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

drop_rate = tf.compat.v1.placeholder(1.0)

#2. 모델

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, rate=drop_rate)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias3')
logits = tf.matmul(layer2, w3) + b3
hypothesis = tf.sigmoid(logits)

#3-1. 컴파일
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
train = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
epochs = 10001
for step in range(epochs):
    cost_val, _, = sess.run([loss, train], 
                                         feed_dict={x:x_data, y:y_data, drop_rate:0.0})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val,)

#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, tf.float32)
pred, acc = sess.run([y_pred, tf.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))],
                         feed_dict={x:x_data, y:y_data, drop_rate:0.0})

from sklearn.metrics import accuracy_score
print("예측:", pred.flatten())
print("정확도:", acc)
