import pandas as pd
from konlpy.tag import Okt
import random
import pickle
from keras.preprocessing import sequence
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def data_load(option):
    csv = pd.read_csv('train_data_add.csv')
    titles_csv = csv['title']
    prices_csv = csv['price']
    if option == "titles" :
        csv = titles_csv
    elif option == 'price' :
        csv = prices_csv
    return csv

def tokenizer_create(text):
    okt = Okt()
    text_pos = okt.pos(text, norm=True)
    
    words = []
    for word in text_pos:
        words.append(word[0])
    
    return words

def words_to_ids(words, word_dict):
    ids = []
    for word in words:
        try:
            ids.append(word_dict.index(word))
        except Exception as e:
            print(e)

    return ids

def dictionary_create():
    data_load()
    okt = Okt()
    words_set = set()
    titles_words = []
    count = 1
    for title in data_load("titles"):
        title_pos = okt.pos(title, norm=True)
        words = []
        for word in title_pos:
            words_set.add(word[0])
            words.append(word[0])
        titles_words.append(words)
        count += 1
        
    dictionary = list(words_set)
    random.shuffle(dictionary)
    dictionary = [0] + dictionary
    
    titles_ids = []
    count = 1
    for title in titles_words:
        words_id = words_to_ids(title, dictionary)
        titles_ids.append(words_id)
        count += 1

def ids_to_words(ids):
    words = []
    for word_id in ids:
        if word_id != 0:
            words.append(dictionary[word_id])
    return words

def sequence_create(text_ids) :
    sequence_np = sequence.pad_sequences([text_ids], maxlen=max_title_len, padding='post')
    return sequence_np

try:
    with open("titles_words.bin", "rb") as f:
        titles_words = pickle.load(f)
    with open("dictionary.bin", "rb") as f:
        dictionary = pickle.load(f)
    with open("titles_ids.bin", "rb") as f:
        titles_ids = pickle.load(f)
        
except Exception as e:
    
    dictionary_create()
        
    with open("titles_words.bin", "wb") as f:
        pickle.dump(titles_words, f)
    with open("dictionary.bin", "wb") as f:
        pickle.dump(dictionary, f)
    with open("titles_ids.bin", "wb") as f:
        pickle.dump(titles_ids, f)


max_title_len = max(len(title_ids) for title_ids in titles_ids)
# print(max_title_len)

titles_ids_np = sequence.pad_sequences(titles_ids, maxlen=max_title_len, padding='post')
# print(titles_ids_np)

prices_np = np.array([[price] for price in data_load("price")])
# print(prices_np)


index = [i for i in range(len(titles_ids_np))]
random.shuffle(index)

train_len = int(len(index) * 0.9)
train_index = index[:train_len]
test_index = index[train_len:]

# print(len(titles_ids_np))
# print(len(train_index))
# print(len(test_index))

X_train = titles_ids_np[train_index]
X_test = titles_ids_np[test_index]

scaler = MinMaxScaler()  # StandardScaler()
scaler.fit(prices_np)
y_scaled = scaler.transform(prices_np)

y_train_scaled = y_scaled[train_index]
y_test_scaled = y_scaled[test_index]

# print(prices_np)
# print(y_scaled)


vocab_size = len(dictionary)


model_name = "baseline_model_data_add.h5"
def model_create() :

    model = keras.Sequential([
        layers.Embedding(vocab_size, 64),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    model.summary()

    model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(1e-4), metrics=['mae'])
    history = model.fit(X_train, y_train_scaled, epochs=10, validation_data=(X_test, y_test_scaled),
                        validation_steps=30, verbose=1)
    model.save(model_name)

try:
    model = models.load_model(model_name)
except Exception as e:
    print(e)
    model_create()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

# plot_graphs(history, 'mae')
# plot_graphs(history, 'loss')

# price_predictions = model.predict(X_test)
#
# y_test_inverse = scaler.inverse_transform(y_test_scaled)
# price_predictions_inverse = scaler.inverse_transform(price_predictions)

# for i in range(100):
#     print(f"{i}: {ids_to_words(X_test[i])}")
#     print(f"{i}: {y_test_inverse[i]} = {price_predictions_inverse[i]}")
#     print()


# print(ids_to_words(X_test[5]))

def predict_phone(text) :
    text_words = tokenizer_create(text)
    print(text_words)
    text_ids = words_to_ids(text_words, dictionary)
    text_ids_np = sequence_create(text_ids)
    text_predictions = model.predict(text_ids_np)
    text_predictions_inverse = scaler.inverse_transform(text_predictions)
    print(text_predictions_inverse)
    return text_predictions_inverse



test1 = "아이폰SE 2세대 128G 59만원에 판매합니다"
test2 = "갤럭시S9 256G 29만원 판매합니다"

predict_phone(test1)
predict_phone(test2)

