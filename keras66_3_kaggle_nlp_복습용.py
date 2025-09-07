# https://www.kaggle.com/competitions/nlp-getting-started/data
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + ' keyword: ' + df['keyword'] + ' location: ' + df['location']
    return df

path = './_data/kaggle/nlp/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submission = pd.read_csv(path+'sample_submission.csv')

train = merge_text(train)
test = merge_text(test)

X = train['text']
y = train['target']
test = test['text']

token = Tokenizer(num_words=10000, oov_token='<OOV>')
token.fit_on_texts(X)

# print(len(token.word_index)) # 24591
# exit()

X = token.texts_to_sequences(X)
test = token.texts_to_sequences(test)

print("최대 길이 : ", max(len(x) for x in X))    # 40
print("최소 길이 : ", min(len(x) for x in X))    # 4
print("평균 길이 : ", sum(map(len, X)) / len(X)) # 21.79

X = pad_sequences(
    X,
    maxlen=40,
    padding='pre',
    truncating='pre'
)

test = pad_sequences(
    test,
    maxlen=40,
    padding='pre',
    truncating='pre'
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=333, stratify=y)

model = Sequential([
            Input(shape=X.shape[1:]),
            Embedding(input_dim=10002, output_dim=256, input_length=40),
            Bidirectional(LSTM(120, activation='tanh', return_sequences=True)),
            Bidirectional(LSTM(60, activation='tanh')),
            Dense(60, activation='relu'),
            Dense(30, activation='relu'),
            Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    validation_data=[X_val, y_val],
    callbacks=[es]
)

y_pred = (model.predict(X_val) >= 0.5).astype(int)
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy : {acc:.6f}") # Accuracy : 0.994747

model.save(path+'lgbm_model.hdf5')
submission['target'] = (model.predict(test) >= 0.5).astype(int)
path = './_save/kaggle/nlp/'
submission.to_csv(path+'submission_2.csv', index=False)
