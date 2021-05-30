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
    if option == "titles":
        csv = titles_csv
    elif option == 'price':
        csv = prices_csv
    return csv


def words_to_ids(words, word_dict):
    ids = []
    for word in words:
        try:
            ids.append(word_dict.index(word))
        except Exception as e:
            print(e)
    return ids


class RNN:

    def __init__(self, model_name):
        self._model_name = model_name

        try:
            with open("titles_words.bin", "rb") as f:
                self._titles_words = pickle.load(f)
            with open("dictionary.bin", "rb") as f:
                self._dictionary = pickle.load(f)
            with open("titles_ids.bin", "rb") as f:
                self._titles_ids = pickle.load(f)
            print("------------Data 사전을 로드합니다--------------")

        except Exception as e:
            print("------------Data 사전이 없으므로 생성합니다----------")
            okt = Okt()
            words_set = set()
            self._titles_words = []
            count = 1
            for title in data_load("titles"):
                title_pos = okt.pos(title, norm=True)
                words = []
                for word in title_pos:
                    words_set.add(word[0])
                    words.append(word[0])
                self._titles_words.append(words)
                count += 1

            dictionary = list(words_set)
            random.shuffle(dictionary)
            self._dictionary = [0] + dictionary

            self._titles_ids = []
            count = 1
            for title in self._titles_words:
                words_id = words_to_ids(title, self._dictionary)
                self._titles_ids.append(words_id)
                count += 1


    def ids_to_words(self, ids):
        words = []
        for word_id in ids:
            if word_id != 0:
                words.append(self._dictionary[word_id])
        return words

    def index_process(self):
        self._max_title_len = max(len(title_ids) for title_ids in self._titles_ids)
        # print(max_title_len)
        titles_ids_np = sequence.pad_sequences(self._titles_ids, maxlen=self._max_title_len, padding='post')
        # print(titles_ids_np)
        self._prices_np = np.array([[price] for price in data_load("price")])
        # print(prices_np)

        index = [i for i in range(len(titles_ids_np))]
        random.shuffle(index)

        train_len = int(len(index) * 0.9)
        train_index = index[:train_len]
        test_index = index[train_len:]

        # print(len(titles_ids_np))
        # print(len(train_index))
        # print(len(test_index))

        self._X_train = titles_ids_np[train_index]
        self._X_test = titles_ids_np[test_index]

        self._scaler = MinMaxScaler()  # StandardScaler()
        self._scaler.fit(self._prices_np)
        y_scaled = self._scaler.transform(self._prices_np)

        self._y_train_scaled = y_scaled[train_index]
        self._y_test_scaled = y_scaled[test_index]

        # print(prices_np)
        # print(y_scaled)

        self._vocab_size = len(self._dictionary)

    @staticmethod
    def tokenizer_create(text):
        okt = Okt()
        text_pos = okt.pos(text, norm=True)

        words = []
        for word in text_pos:
            words.append(word[0])

        return words

    def sequence_create(self, text_ids):
        sequence_np = sequence.pad_sequences([text_ids], maxlen=self._max_title_len, padding='post')
        return sequence_np

    def model_create(self):
        model = keras.Sequential([
            layers.Embedding(self._vocab_size, 64),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])
        model.summary()

        model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(1e-4), metrics=['mae'])
        history = model.fit(self._X_train, self._y_train_scaled, epochs=1,
                            validation_data=(self._X_test, self._y_test_scaled),
                            validation_steps=30, verbose=1)
        model.save(self._model_name)
        with open("titles_words.bin", "wb") as f:
            pickle.dump(self._titles_words, f)
        with open("dictionary.bin", "wb") as f:
            pickle.dump(self._dictionary, f)
        with open("titles_ids.bin", "wb") as f:
            pickle.dump(self._titles_ids, f)
        return model

    def model_load(self):
        try:
            model = models.load_model(self._model_name)
            print("-----------RNN 모델을 로드합니다-------------")
            return model
        except Exception as e:
            print(e)
            print("-----------RNN 모델이 없으므로 학습을 진행합니다-------------")
            model = self.model_create()
            return model

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

    def predict_phone(self, text):
        text_words = self.tokenizer_create(text)
        print(text_words)
        text_ids = words_to_ids(text_words, self._dictionary)
        text_ids_np = self.sequence_create(text_ids)
        model = self.model_load()
        predictions = model.predict(text_ids_np)
        text_predictions_inverse = self._scaler.inverse_transform(predictions)
        print(f'{text} -> {text_predictions_inverse}')
        return text_predictions_inverse


test1 = "아이폰SE 2세대 128G 신상 급처 합니다"
test2 = "갤럭시S10 256G 중고가로 판매합니다"
test3 = "갤럭시A6 32G 급처합니다"
try:
    a = RNN("baseline_model_data_add.h5")
    a.index_process()
except Exception as e:
    print(e)

a.predict_phone(test1)
a.predict_phone(test2)
a.predict_phone(test3)
