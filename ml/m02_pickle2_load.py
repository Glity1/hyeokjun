import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

# 2. 모델 구성
# 3. 훈련(불러오기)
path = './_save/m01_job/'
model = pickle.load(open(path+'m02_pickle_save.pkl', 'rb'))

# 4. 평가, 예측
results = model.score(x_test, y_test) # accuracy score
print(f"Score : {results}")

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)    
print(f"Accuracy score : {acc}")
