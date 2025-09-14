from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

parameters = {
    'n_estimators' : 1000,
    'learning_rate' : 0.3,
    'max_depth' : 3, 
    'gamma' : 1,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
    'verbose' : 0,
}

# 2. 모델 구성
model = XGBClassifier(
    # **parameters
)

# 3. 훈련
model.set_params(**parameters)

model.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    verbose=10
)

# 4. 평가, 예측
results = model.score(x_test, y_test) # accuracy score
print(f"Score : {results}")

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score : {acc}")

path = './_save/m01_job/'
model.save_model(path+'m03_xgb_save.ubj')
