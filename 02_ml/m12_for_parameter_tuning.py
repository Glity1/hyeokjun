import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=55
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3,4,5,6,7]

best_score = 0
best_parameters = ''

for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(learning_rate=lr, max_depth=md)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_test)
        
        if best_score<score:
            best_score = score
            best_parameters = lr, md
        print(i+1, ',', j+1, '번째 도는중... \nscore : ', np.round(score, 3))
        print('최고점 : ', np.round(best_score, 3))
        

print('최고점수 : {:.2f}'.format(best_score))
print('최적 매개변수 : ', best_parameters)

# 최고점수 : 0.98
# 최적 매개변수 :  (0.1, 6)