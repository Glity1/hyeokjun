# 34_2 copy
# LDA n_component는 y 라벨의 갯수 -1 이하로 만들 수 있다.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,
    stratify=y
)

######### pca하기전에 scaler하는게 좋다 ###############
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
print(x.shape) # (150, 4)

exit()

#2. 모델
model = RandomForestClassifier(random_state=222)

for i in range(1,3) : 
    lda = LinearDiscriminantAnalysis(n_components=i)                     
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)
    
    model.fit(x_train_lda, y_train)
    score = model.score(x_test_lda, y_test)
    print(f"[LDA n_components={i}] Test Score: {score:.4f}")
    
    
# [PCA n_components=1] Test Score: 0.9333
# [PCA n_components=2] Test Score: 0.8667
# [PCA n_components=3] Test Score: 0.9667
# [PCA n_components=4] Test Score: 0.9333
























































