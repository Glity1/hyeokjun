import tensorflow as tf

tf.random.set_random_seed(518)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')
# <tf.Variable 'weights:0' shape=(2,) dtype=float32_ref>

print(변수)
### [변수 초기화] ###
#1. global_variables_initializer
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa :', aaa) # aaa : [ 1.0413485 -0.7816172]
sess.close()

#2. 변수.eval
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) # Tensorflow 데이터형태인 '변수'를 Python에서 사용가능하게 변경
print('bbb :', bbb)
sess.close()

#3. tf.compat.v1.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()    # eval에 default로 들어가는 놈
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc :", ccc)
sess.close()