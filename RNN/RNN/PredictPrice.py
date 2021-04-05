import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from IPython.display import clear_output
from gensim.models.word2vec import Word2Vec


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

csv = pd.read_csv('train_data.csv')
csv.info()
titles = csv['title']
prices = csv['price']

okt = Okt()

splited_titles = []
count = 0
for title in titles:
    morphemes = []
    title_pos = okt.pos(title, norm=True)
    for morpheme in title_pos:
        morphemes.append(morpheme[0])
    splited_titles.append(morphemes)
    count += 1
    clear_output(wait=True)
    print(f"{count} / {len(titles)}")

model_wv = Word2Vec(sentences = splited_titles, min_count = 1)
model_wv.save("word2vec.model")

train_titles = []
count = 0
for title in splited_titles:
    vectors = []
    for morpheme in title:
        vectors.append(model_wv.wv[morpheme])
    train_titles.append(vectors)
    count += 1
    clear_output(wait=True)
    print(f"{count} / {len(splited_titles)}")


trainData, testData, trainResult, testResult = train_test_split(train_titles, prices)
print(len(trainData))
print(len(testData))
print(len(trainResult))
print(len(testResult))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())


history = model.fit(trainData, epochs=1,
                    validation_data=testData,
                    validation_steps=30,
                    verbose=1)

