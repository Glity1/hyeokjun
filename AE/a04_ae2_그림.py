import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA

# 1) 데이터 & 노이즈
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test  = x_test.reshape(-1, 28*28).astype('float32')/255.

rng = np.random.default_rng(42)
x_train_noised = np.clip(x_train + rng.normal(0, 0.1, x_train.shape), 0, 1)
x_test_noised  = np.clip(x_test  + rng.normal(0, 0.1, x_test.shape),  0, 1)

# 2) PCA로 적정 차원 보기
pca = PCA(n_components=784).fit(x_train)  # 누적 설명분산만 확인용
evr_cumsum = np.cumsum(pca.explained_variance_ratio_)
k95  = np.searchsorted(evr_cumsum, 0.95) + 1
k99  = np.searchsorted(evr_cumsum, 0.99) + 1
k999 = np.searchsorted(evr_cumsum, 0.999) + 1
k100 = 784  # 전부

print(f"0.95 이상: {k95}, 0.99 이상: {k99}, 0.999 이상: {k999}, 1.0: {k100}")

# 3) 오토인코더 (denoising: noisy -> clean)
hidden_size = k95  # 예: 95% 분산 유지 수준으로 병목 설정
model = Sequential([
    Dense(hidden_size, input_shape=(784,), activation='relu'),
    Dense(784, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

# 4) 평가/시각화
decoded = model.predict(x_test_noised, verbose=0)

fig, axes = plt.subplots(3, 5, figsize=(12, 7))
idxs = random.sample(range(len(x_test)), 5)
for i, idx in enumerate(idxs):
    axes[0, i].imshow(x_test[idx].reshape(28, 28), cmap='gray')        # 원본
    axes[1, i].imshow(x_test_noised[idx].reshape(28, 28), cmap='gray') # 노이즈
    axes[2, i].imshow(decoded[idx].reshape(28, 28), cmap='gray')       # 복원
    for r in range(3):
        axes[r, i].axis('off')
axes[0,0].set_ylabel('Clean', rotation=0, labelpad=30, va='center')
axes[1,0].set_ylabel('Noisy', rotation=0, labelpad=30, va='center')
axes[2,0].set_ylabel('Denoised', rotation=0, labelpad=30, va='center')
plt.tight_layout()
plt.show()
