import numpy as np                                           
import pandas as pd 
import tensorflow as tf                                         
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

#1. 데이터
path = './_data/dacon/diabetes/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome'] 
x = x.replace(0, np.nan)                
x = x.fillna(x.mean())
test_csv = test_csv.fillna(x.mean())


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True,
    )

# print(x_train.shape, x_test.shape)  #(521, 8) (131, 8)
# print(y_train.shape, y_test.shape)  #(521,)  (131,)

y_train = y_train.values.reshape(-1, 1).astype(np.float32)
y_test  = y_test.values.reshape(-1, 1).astype(np.float32)
x_train = x_train.values.astype(np.float32)
x_test  = x_test.values.astype(np.float32)

# 2. placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), name='weights')
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


# 최종 Test Accuracy: 0.7633587786259542