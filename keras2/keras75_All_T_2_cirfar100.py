import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
    ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    NASNetMobile, EfficientNetB0, EfficientNetB1
)

# 1. 데이터 로드 및 전처리
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# 2. 사용할 사전학습 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(32, 32, 3)),
    VGG19(include_top=False, input_shape=(32, 32, 3)),
    ResNet50(include_top=False, input_shape=(32, 32, 3)),
    ResNet50V2(include_top=False, input_shape=(32, 32, 3)),
    ResNet101(include_top=False, input_shape=(32, 32, 3)),
    ResNet101V2(include_top=False, input_shape=(32, 32, 3)),
    ResNet152(include_top=False, input_shape=(32, 32, 3)),
    ResNet152V2(include_top=False, input_shape=(32, 32, 3)),
    DenseNet121(include_top=False, input_shape=(32, 32, 3)),
    DenseNet169(include_top=False, input_shape=(32, 32, 3)),
    DenseNet201(include_top=False, input_shape=(32, 32, 3)),
    MobileNet(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV3Small(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV3Large(include_top=False, input_shape=(32, 32, 3)),
    # NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB1(include_top=False, input_shape=(32, 32, 3)),
]

for model in model_list:
    model.trainable = True

    print("===================================================")
    print("모델명 :", model.name)
    print("전체 가중치 갯수 :", len(model.weights))
    print("훈련 가능 가중치 갯수 :", len(model.trainable_weights))

# 3. 모델별 반복 학습 및 평가
for base_model in model_list:
    try:
        base_model.trainable = True

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(100, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        print(f"\n🚀 Training model: {base_model.name}")
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es], verbose=0)

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ {base_model.name} | val_loss: {loss:.4f} | val_acc: {acc:.4f}")

        # accuracy_score로도 출력
        y_pred = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test.values, axis=1)
        acc_score = accuracy_score(y_true, y_pred)
        print(f"📊 {base_model.name} | accuracy_score: {acc_score:.4f}")

    except Exception as e:
        print(f"❌ {base_model.name} failed: {e}")
