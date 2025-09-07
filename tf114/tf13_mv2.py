import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(14)

#1. 데이터
# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]
# y_data = [152., 185., 180., 196., 142.]

x_data = [[77, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [73, 66, 70],
          ] #(5,3)

y_data = [[152.], [185.], [180.], [196.], [142.]]

# x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])  # tf.compat.v1.placeholder는 TensorFlow 1.x 스타일 코드에서만 필요한 함수입니다.
# x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
# x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y = tf.compat.v1.placeholder(tf.float32, shape=[None])

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='bias')

#2. 모델구성
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬 연산처리를 위해서

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련

epochs = 6000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],feed_dict={x:x_data, y:y_data})

    print(step, '\t', cost_val)

sess.close()


# 5953     0.11563341