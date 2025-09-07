import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(777)

import matplotlib.pyplot as plt

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random.normal([1], dtype=tf.float32))
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])


#2. 모델구성
hypothesis = x* w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1000
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

    print("===========================predict================================")
    y_predict = x_test * w_val + b_val
    print("[6,7,8]결과: ", sess.run(y_predict, feed_dict={x_test:x_test_data}))

print("===================그림 그리기=====================")
# print(loss_val_list)
# print(w_val_list)

# loss 와 epoch 관계
# plt.plot(loss_val_list)
# plt.show()

# w 와 epoch 관계
# plt.plot(w_val_list)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('weight')
# plt.show()

# w 와 loss의 관계
# plt.plot(w_val_list, loss_val_list)
# plt.grid()
# plt.xlabel('weigth')
# plt.ylabel('loss')
# plt.show()

#[실습] subplot으로 위 3개의 그래프를 한페이지에 나오게 수정
# plt.subplot(221)
# plt.plot(w_val_list, loss_val_list)
# plt.grid()
# plt.xlabel('weigth')
# plt.ylabel('loss')

# plt.subplot(222)
# plt.plot(w_val_list)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('weight')

# plt.subplot(223)
# plt.plot(loss_val_list)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# (1) Weight vs Loss
plt.subplot(221)
plt.plot(w_val_list, loss_val_list, color="royalblue", linewidth=2, marker="o", markersize=4)
plt.title("Weight vs Loss", fontsize=14, fontweight="bold")
plt.xlabel("Weight", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# (2) Epoch vs Weight
plt.subplot(222)
plt.plot(w_val_list, color="darkorange", linewidth=2)
plt.title("Epoch vs Weight", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Weight", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# (3) Loss over Epochs
plt.subplot(223)
plt.plot(loss_val_list, color="seagreen", linewidth=2)
plt.title("Epoch vs Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()