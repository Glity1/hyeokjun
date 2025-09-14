"""
06. cancer
07. dacon_diabetes
08. kaggle_bank

09. wine
10. fetch_covtype

11. digits
12. kaggle_santander
13. kaggle_otto
"""

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     
from sklearn.tree import DecisionTreeClassifier         
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
data_list = [
    ("iris", load_iris(return_X_y=True)),
    ("breast_cancer", load_breast_cancer(return_X_y=True)),
    ("digits", load_digits(return_X_y=True)),
    ("wine", load_wine(return_X_y=True)),
]

model_list =  [
    LinearSVC(C=0.3, max_iter=10000),
    LogisticRegression(max_iter=10000),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=100),
]

for data_name, (x, y) in data_list:
    print(f" 데이터리스트: {data_name}")
    for model in model_list:
        model_name = model.__class__.__name__
        model.fit(x, y)
        results = model.score(x, y)
        print(f"{model_name} 정확도: {results:.4f}")  # f를 붙이면 문자열 안에 변수 값을 직접 넣을 수 있다! / 문자열 안에 {}를 통해 변수 삽입 가능하게 함
        
# 처음 iris를 4개의 모델을 통해 학습시키더라도 다음 for문에서 cancer을 학습시키면 iris에 대한 기억은 삭제된다.

# 데이터리스트: iris
# LinearSVC 정확도: 0.9600
# LogisticRegression 정확도: 0.9733
# DecisionTreeClassifier 정확도: 1.0000
# RandomForestClassifier 정확도: 1.0000

# 데이터리스트: breast_cancer
# LinearSVC 정확도: 0.9367
# LogisticRegression 정확도: 0.9578
# DecisionTreeClassifier 정확도: 0.9947
# RandomForestClassifier 정확도: 1.0000

# 데이터리스트: digits
# LinearSVC 정확도: 0.9961
# LogisticRegression 정확도: 1.0000
# DecisionTreeClassifier 정확도: 0.7073
# RandomForestClassifier 정확도: 1.0000

# 데이터리스트: wine
# LinearSVC 정확도: 0.9157
# LogisticRegression 정확도: 0.9944
# DecisionTreeClassifier 정확도: 1.0000
# RandomForestClassifier 정확도: 1.0000