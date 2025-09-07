import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

sess = tf.compat.v1.Session()

### [실습] ###
# 덧셈    add
add = tf.add(node1, node2)

# 뺄셈    substract
sub = tf.subtract(node1, node2)

# 곱셈    multiply
mtp = tf.multiply(node1, node2)

# 나눗셈  divide
dvd = tf.divide(node1, node2)

print(sess.run(add))     # 7.0   
print(sess.run(sub))     # -1.0
print(sess.run(mtp))     # 12.0
print(sess.run(dvd))     # 0.75