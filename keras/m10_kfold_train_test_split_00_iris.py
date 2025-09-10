import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=333, shuffle=True,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True)

#2. 모델
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc : ', np.round(np.mean(scores), 4))

# acc :  [0.91666667 1.         0.95833333 1.         0.95833333] 
# 평균 acc :  0.9667

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_pred)

# [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1]
# [2 0 2 2 2 1 1 0 2 0 0 0 0 2 2 2 0 2 0 1 1 0 1 1 2 0 1 1 1 1]

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

# cross_val_predict ACC :  0.8666666666666667