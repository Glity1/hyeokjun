# 주성분 분석
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (150, 4) (150,)

######### pca하기전에 scaler하는게 좋다 ###############
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=3)                     # n_components 를 얼마로 잡은것인지 고민하면된다. // 너무 낮으면 성능이 떨어짐.
x = pca.fit_transform(x)
print(x)
print(x.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
    stratify=y
)

#2. 모델
model = RandomForestClassifier(random_state=222)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print(x.shape, '의 score : ', results) 
























































