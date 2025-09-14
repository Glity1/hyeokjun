import tensorflow as tf

print(tf.__version__)

## tensorflow 1.14 설치시 오류
# protobuf version : 실행 불가 > 3.20
# numpy version : warning > 1.16

print('hello world')     # hello world

hello = tf.constant('hello world')
print(hello)             # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))   # b'hello world'
                         # b = binary는 호남선 , 문자

# python : input > output
# tensor : input > Tensor Machine > outPut
               # TM : tensor 형태의 데이터를 연산해주는 툴
               # 실행 명령어 = sess.run : 그래프 연산을 실행시킴


