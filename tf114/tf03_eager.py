import tensorflow as tf

print('텐서버전 :', tf.__version__)             # 1.14.0
print('즉시실행 :', tf.executing_eagerly())     # False

# 텐서버전 : 2.15.0
# 즉시실행 : True

tf.compat.v1.disable_eager_execution()
print('즉시실행 :', tf.executing_eagerly())     # False

tf.compat.v1.enable_eager_execution()
print('즉시실행 :', tf.executing_eagerly())     # True

# eager을 껐다 켰다하고 있는데 이건 무엇인가??
# 즉시 실행(sess.run 생략) 모드를 껐다 켰다 하는것

hello = tf.constant("hello wordl!!")
sess = tf.compat.v1.Session()

print(sess.run(hello))

# [SUMMARY] #
# 즉시 실행 모드 > tensor1의 그래프 형태의 구성 없이 자연스러운 python 문법으로 실행
# tf.compat.v1.disable_eager_execution() > 즉시 실행 off : tensor 1.0 문법 사용 가능
# tf.compat.v1.enable_eager_execution() > 즉시 실행 on : tensor 2.0 문법 사용 가능

# sess.run() 실행시...
# 가상환경          즉시실행모드            사용가능
# 1.14.0           disable (default)      b'Hello world'
# 1.14.0           enable                 error
# 2.7.4            disable (default)      b'Hello world'
# 2.7.4            enable                 error

'''
Tensor1 : 그래프 연산 모드
        : tf.compat.v1.disable_eager_execution()
Tensor2 : 즉시 실행 모드
        : tf.compat.v1.enable_eager_execution()

tf.executing_eargerly()
> True  : 즉시 실행모드, Tensor2 코드에만 사용
> False : 그래프 연산모드, Tensor1 코드를 쓸 수 있음
'''