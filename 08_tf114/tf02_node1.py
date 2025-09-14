import tensorflow as tf

# 3 + 4 = ?

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# node3 = node1 + node2

node3 = tf.add(node1, node2)
# print(node3)   Tensor("Add:0", shape=(), dtype=float32)
               # sess.run이 없으면 Tensor 형태의 그래프 데이터
               # Tensor("함수", 데이터 형태, 데이터 유형)
               
sess = tf.compat.v1.Session()
print(sess.run(node3))     # 7.0