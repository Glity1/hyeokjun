import numpy as np                                           
import pandas as pd                                          
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization
import seaborn as sns

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
    random_state=39,
    shuffle=True,
    stratify=y
)

# 스케일링
scaler = MinMaxScaler()
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
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

path = './_save/keras23/'
model.save(path + 'keras23_1_save.h5')


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)
hist = model.fit(
    x_train, y_train, 
    epochs=400, batch_size=128,
    validation_split=0.2,
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
submission_csv.to_csv(path + 'submission_0613_1800.csv')

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

# 1. Confusion Matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc

cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. Precision-Recall Curve (이미 일부 구했지만 다시 보기 쉽게 시각화)
plt.figure(figsize=(7, 5))
plt.plot(recall, prec, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 4. 예측 확률 분포 (Positive / Negative 분포)
plt.figure(figsize=(8, 5))
sns.histplot(y_proba[y_test==0], color='skyblue', label='Negative', stat='density', kde=True)
sns.histplot(y_proba[y_test==1], color='salmon', label='Positive', stat='density', kde=True)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
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

loss :  0.5466394424438477
acc :  0.883662223815918
Best threshold for F1: 0.6098
Best F1 score on validation: 0.4895
f1_score :  0.48949633004302706
acc_score :  0.8842932537861404

loss :  0.548361599445343
acc :  0.8867025971412659
Best threshold for F1: 0.7698
Best F1 score on validation: 0.4973
f1_score :  0.4973166368515206
acc_score :  0.8871615419917394

"""