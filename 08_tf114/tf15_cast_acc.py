import tensorflow as tf

tf.compat.v1.disable_eager_execution()

aaa = tf.constant([0.3, 0.4, 0.8, 0.9])
bbb = tf.constant([0, 1, 1, 1], dtype=tf.float32)

sess = tf.compat.v1.Session()

pred = tf.cast(aaa > 0.5, dtype=tf.float32)
predict = sess.run(pred)
print(predict) # [0. 0. 1. 1.]

acc = tf.reduce_mean(tf.cast(tf.equal(pred, bbb), dtype=tf.float32))
print(sess.run(acc)) # 0.75

sess.close()