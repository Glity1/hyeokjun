import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')   # 현재 tensorflw cpu버전을 설치했기 때문에 GPU 버전으로 새로 설치가 필요함.
    
    
