import tensorflow as tf

print(tf.__version__)

# node1 = tf.constant(3.0)
# node2 = tf.constant(4.0)
# node3 = node1 + node2

# 그래프 구성
node1 = tf.compat.v1.placeholder(tf.float32)
node2 = tf.compat.v1.placeholder(tf.float32)
node3 = node1 + node2
node3_triple = node3 * 3
# placeholder : 상수 입력에 특화된 것

# Tensor Machine으로 입력
sess = tf.compat.v1.Session()

# 구성된 그래프에 변수 입력 
print(sess.run(node3,
               feed_dict = {node1:3,
                            node2:4}))
# print(sess.run(node3,
#                feed_dict = {node1:10,
#                             node2:17}))

print(sess.run(node3_triple,
               feed_dict = {node1:3,
                            node2:4}))