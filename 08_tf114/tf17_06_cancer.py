import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

# 1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data.astype(np.float32)
y_data = datasets.target.reshape(-1, 1).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=384, shuffle=True
)

# 2. placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

# 3. 모델
logits = tf.matmul(x, w) + b
hypothesis = tf.sigmoid(logits)


# 4. 손실 함수 (내장 함수 사용 → 안정적)
loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
)

# 5. 최적화기 (Adam 권장)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 예측, 정확도
pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

# 6. 학습
epochs = 2001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, cost_val, train_acc = sess.run([train, loss, acc],
                                feed_dict={x: x_train, y: y_train})
        if step % 200 == 0:
            print(f"Epoch {step:5d} | Loss: {cost_val:.4f} | Train Acc: {train_acc:.4f}")

    # 테스트 예측
    y_predict = sess.run(pred, feed_dict={x: x_test})
    final_acc = accuracy_score(y_test, y_predict)

    print("최종 Test Accuracy:", final_acc)

# 최종 Test Accuracy: 0.9707602339181286