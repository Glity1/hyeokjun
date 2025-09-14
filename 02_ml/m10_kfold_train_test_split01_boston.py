import numpy as np
import pandas as pd
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 7
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)                   # StratifiedKFold 범주형일 때 확실하게 분류 // 데이터가 많으면 kfold써도됨.

#2. 모델구성
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

# acc :  [0.83551057 0.84259202 0.93069851 0.82530124 0.88683223 0.85195263 0.90458656]
#  평균 acc :  0.8682

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
# y_pred = np.round(y_pred)
# print(y_test) #[1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]                 
# print(y_pred) #[1 0 2 2 0 0 1 1 2 0 0 2 1 1 1 1 0 0 0 0 0 2 2 2 1 1 1 1 1 1]                 # round처리 자동으로 해줌

acc = r2_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc) # cross_val_predict ACC :  0.5869984238086257


