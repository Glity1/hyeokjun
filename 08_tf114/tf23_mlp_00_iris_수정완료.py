#20-1 copy
import tensorflow as tf
from sklearn.datasets import load_iris                         #3가지 품종을 맞출려고한다. 0,1,2 // 다중분류                                                      
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)

#1. 데이터
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.1, random_state=731,
)

drop_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

#2. 모델

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([10]), name='bias2')
layer2 = tf.matmul(layer1, w2) + b2
layer2 = tf.nn.dropout(layer2, rate=drop_rate)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,3]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([3]), name='bias3')
logits = tf.matmul(layer2, w3) + b3
hypothesis = tf.nn.softmax(logits)

#3-1. 컴파일
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
epochs = 10001
for step in range(epochs):
    cost_val, _, = sess.run([loss, train], 
                                         feed_dict={x:x_data, y:y_data, drop_rate:0.3})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val,)

#4. 평가, 예측
y_pred = tf.argmax(hypothesis, axis=1)
y_true = tf.argmax(y, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

print("정확도:", acc)
