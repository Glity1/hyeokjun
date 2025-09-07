#20-1 copy
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# reshape target
y_train = y_train.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)



drop_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

#2. 모델

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([10]), name='bias2')
layer2 = tf.matmul(layer1, w2) + b2
layer2 = tf.nn.dropout(layer2, rate=drop_rate)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias3')
hypothesis = tf.matmul(layer2, w3) + b3

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
epochs = 10001
for step in range(epochs):
    cost_val, _, = sess.run([loss, train], 
                                         feed_dict={x:x_train, y:y_train, drop_rate:0.3})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val,)

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test, drop_rate: 0.0})
sess.close()

# R2, MAE 계산
r2 = r2_score(y_test, y_predict)   # 실제값 vs 훈련데이터 예측값
mse = mean_squared_error(y_test, y_predict)

print('r2_score : ', r2) 
print('mse : ', mse)

