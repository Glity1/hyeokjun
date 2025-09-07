import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()

sess.run(init)
print(sess.run(a + b))

# 상수 : 정의가 반드시 필요
# 변수 : 초기화가 반드시 필요