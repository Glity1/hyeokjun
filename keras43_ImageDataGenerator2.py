import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation

# train 160   test 120 
 
train_datagen = ImageDataGenerator(
    # ImageDataGenerator 이미지 데이터를 수치화 시켜준다. 
    rescale=1./255,                          # 0~255로 스케일링, 정규화                       
    horizontal_flip=True,                    # 수평 뒤집기 <- 데이터 증폭 또는 변환
    vertical_flip=True,                      # 수직 뒤집기 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1,                   # 평행이동 10% 
    # height_shift_range=0.1,                   
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,                       # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환 (끌어당기거나 짜부시킴) 
    fill_mode='nearest'                      # 특정지역에 값이 소멸되는 곳에 대체값으로 근사값으로 넣어준다)
)                                            # 데이터 증폭
 

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train'
path_test = './_data/image/brain/test'

xy_train = train_datagen.flow_from_directory(            # directory : 폴더
    path_train,
    target_size=(200, 200),                              # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=160,                                       # batch_size : default 32
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    seed=333,
)           

# Found 160 images belonging to 2 classes.  

xy_test = test_datagen.flow_from_directory(             # directory : 폴더
    path_test,
    target_size=(200, 200),                             # 사이즈 원하는대로 조절 가능 (큰 사이즈는 축소, 작은 사이즈는 확대)
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale'
)      

#Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(y_train[1])

# plt.imshow(x_train[1],'gray')
# plt.show()

# exit()

print(x_train.shape, y_train.shape)  # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)    # (120, 200, 200, 1) (120,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(200, 200, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (2,2),))
model.add(Activation('relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])

es = EarlyStopping(                    # EarlyStopping을 es라는 변수에 넣는다
    # monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=30,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,         # 가장 최소 지점으로 저장한다
) 
start = time.time()
hist = model.fit(x_train,y_train, epochs=300,
          verbose =1,
          validation_split=0.2,
          callbacks=[es],
          )

end = time.time()
print("걸린시간 : ", end - start, '초')

print('========================hist============================')
print(hist) # keras.callbacks.History object at 0x000002AE4E644220> 으로 나오는데 제대로 볼려면 
print('========================hist.history============================')
print(hist.history)  # 중괄호의 등장 : 키(loss, val_loss) : 벨류(숫자) 형태로 안에 넣어둔다 // loss, val loss 의 갯수는 epochs 값과 똑같음
                     # loss들의 역사 
                     # 그래프의 시각화가 가능하다 점들의 값이 있기 떄문에

print('========================hist.history에서 loss만 따로보고싶다============================')
print(hist.history['loss'])   # dictionary의 키값만 적어주면된다                     
       
print('========================hist.history에서 val_loss만 따로보고싶다============================')
print(hist.history['val_loss'])   # dictionary의 키값만 적어주면된다

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)

print("loss : ", loss[0])
print("accuray : ", loss[1])

