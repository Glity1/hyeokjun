import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

datasets = load_wine()
x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=814,
    shuffle=True, 
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (142, 13) (36, 13) 
print(y_train.shape, y_test.shape) # (142, 3) (36, 3)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,3]), name = 'weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name = 'bias', dtype=tf.float32)

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)


#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

pred_op = tf.argmax(hypothesis, axis=1)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
epochs = 1001
for step in range(epochs) : 
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b, ], feed_dict={x:x_train, y:y_train})
    if step % 50 ==0:
        print(step, 'loss : ', cost_val, )

print('최종 weight : ', w_val, 'bias : ', b_val)

y_true = np.argmax(np.array(y_test), axis=1)
y_pred = sess.run(pred_op, feed_dict={x: x_test}) 
acc = accuracy_score(y_true, y_pred)  

print("최종 Test Accuracy:", acc)
# 최종 Test Accuracy: 1.0
