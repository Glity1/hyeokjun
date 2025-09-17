import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization

#1. 데이터
path = './_data/dacon/cancer/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# Label Encoding (Country, Race 포함)
le = LabelEncoder()
for col in ['Gender','Country','Race','Family_Background','Radiation_History','Iodine_Deficiency','Smoke','Weight_Risk','Diabetes']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col]  = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer'] 

# train_test_split 시 stratify=y 적용 (클래스 분포 유지)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=352,
    shuffle=True,
    stratify=y
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# 클래스 가중치 계산 (훈련 데이터 기준)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=35,
    restore_best_weights=True,
)
hist = model.fit(
    x_train, y_train, 
    epochs=400, batch_size=256,
    validation_split=0.1,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

#4. 평가 및 최적 threshold 찾기
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

y_proba = model.predict(x_test).ravel()
prec, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold for F1: {best_threshold:.4f}")
print(f"Best F1 score on validation: {f1_scores[best_idx]:.4f}")

y_predict = (y_proba >= best_threshold).astype(int)

f1 = f1_score(y_test, y_predict)
print("f1_score : ", f1)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)

#5. 테스트셋 예측 및 제출
test_pred_proba = model.predict(test_csv_scaled).ravel()
test_pred = (test_pred_proba >= best_threshold).astype(int)

submission_csv['Cancer'] = test_pred
submission_csv.to_csv(path + 'submission_optimized5.csv')

#6. 시각화
import matplotlib.pyplot as plt        
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = "C:/Windows/Fonts/malgun.ttf"  
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))  
plt.plot(hist.history['loss'], c='red', label='loss')                        
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')              
plt.title('갑상선암 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  
plt.grid(True)                     
plt.show()


"""
loss :  0.5329536199569702
acc :  0.8804497718811035
Best threshold for F1: 0.6629
Best F1 score on validation: 0.4889
f1_score :  0.488855116514691
acc_score :  0.8842358880220285


loss :  0.5412371158599854
acc :  0.8709843754768372
Best threshold for F1: 0.6039
Best F1 score on validation: 0.4758
f1_score :  0.4758269720101781
acc_score :  0.8818265259293254     


Best threshold for F1: 0.4455
Best F1 score on validation: 0.4767
f1_score :  0.47672348003052656
acc_score :  0.8819986232216613

"""