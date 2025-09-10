# https://www.kaggle.com/competitions/nlp-getting-started/data
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    return df

path = './_data/kaggle/nlp/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submission = pd.read_csv(path+'sample_submission.csv')

train = merge_text(train)
test = merge_text(test)

X1 = train['text']
X2 = train['keyword']
X3 = train['location']

y = train['target']

test1 = test['text']
test2 = test['keyword']
test3 = test['location']

ohe1 = OneHotEncoder(sparse=False, handle_unknown='ignore')
X2 = ohe1.fit_transform(np.array(X2).reshape(-1,1))
test2 = ohe1.transform(np.array(test2).reshape(-1,1))

ohe2 = OneHotEncoder(sparse=False, handle_unknown='ignore')  # scikit-learn 버전오류 ohe 1을 ohe 2로 바꿈
X3 = ohe2.fit_transform(np.array(X3).reshape(-1,1))
test3 = ohe2.transform(np.array(test3).reshape(-1,1))

token = Tokenizer(num_words=10000, oov_token='<OOV>')
token.fit_on_texts(X1)

# print(len(token.word_index)) # 24591
# exit()

X1 = token.texts_to_sequences(X1)
test1 = token.texts_to_sequences(test1)

print("최대 길이 : ", max(len(x) for x in X1))    # 40
print("최소 길이 : ", min(len(x) for x in X1))    # 4
print("평균 길이 : ", sum(map(len, X1)) / len(X1)) # 21.79

X1 = pad_sequences(
    X1,
    maxlen=40,
    padding='pre',
    truncating='pre'
)

test1 = pad_sequences(
    test1,
    maxlen=40,
    padding='pre',
    truncating='pre'
)

X1_train, X1_val, X2_train, X2_val, X3_train, X3_val, y_train, y_val = train_test_split(X1, X2, X3, y, test_size=0.2, random_state=333, stratify=y)

input1 = Input(shape=X1_train.shape[1:])
embed1 = Embedding(input_dim=10002, output_dim=256, input_length=40)(input1)
dropout1 = Dropout(0.3)(embed1)
lstm1 = Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True))(embed1)
lstm2 = Bidirectional(LSTM(units=32, activation='tanh'))(lstm1)
dense11 = Dense(units=64, activation='relu')(lstm2)
dense12 = Dense(units=32, activation='relu')(dense11)

input2 = Input(shape=X2_train.shape[1:])
dense21 = Dense(units=64, activation='relu')(input2)
dense22 = Dense(units=32, activation='relu')(dense21)
dense23 = Dense(units=16, activation='relu')(dense22)

input3 = Input(shape=X3_train.shape[1:])
dense31 = Dense(units=64, activation='relu')(input3)
dense32 = Dense(units=32, activation='relu')(dense31)
dense33 = Dense(units=16, activation='relu')(dense32)

concat1 = concatenate([dense12, dense23, dense33])
dense41 = Dense(units=64, activation='relu')(concat1)
dense42 = Dense(units=32, activation='relu')(dense41)
output = Dense(units=1, activation='sigmoid')(dense42)

model = Model(inputs=[input1, input2, input3], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    [X1_train, X2_train, X3_train],
    y_train,
    epochs=50,
    batch_size=256,
    validation_data=[[X1_val, X2_val, X3_val], y_val],
    callbacks=[es]
)

y_pred = (model.predict([X1_val, X2_val, X3_val]) >= 0.5).astype(int)
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy : {acc:.6f}") # Accuracy : 0.804990

model.save(path+'lgbm_model_2.hdf5')
submission['target'] = (model.predict([test1, test2, test3]) >= 0.5).astype(int)
path = './_save/kaggle/nlp/'
submission.to_csv(path+'submission_3.csv', index=False)
