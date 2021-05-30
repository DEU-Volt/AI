from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def preprocessing(data):
    indexs = data.index.values
    print(len(indexs))
    print(indexs[:10], indexs[-10:])

def data_load():
    for cnt, chunck in enumerate(pd.read_csv('train_data_add.csv', chunksize=10 ** 1)) :
        #preprocessing(chunck)
        if cnt >=10 :
            break
        df = chunck['title']
    data = df.to_numpy()
    print(data.shape)
    return data

x_data = data_load()
text = input("타이틀을 입력하세요")

okt = Okt()
for i, document in enumerate(x_data):
    nouns = okt.nouns(document)
    x_data[i] = ' '.join(nouns)
print(x_data)

vect = TfidfVectorizer()

x_data = vect.fit_transform(x_data)

cosine_similarity_matrix = (x_data * x_data.T)
print(cosine_similarity_matrix.shape)
print(cosine_similarity_matrix.toarray())

sns.heatmap(cosine_similarity_matrix.toarray(), cmap='viridis')
plt.show()
