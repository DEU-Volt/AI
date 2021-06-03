from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from konlpy.tag import Okt


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def top5_indices(text, data, q_num):
    cos_sim = linear_kernel(text, data)

    cos_sim_score = list(enumerate(cos_sim[q_num]))
    cos_sim_score = sorted(cos_sim_score, key=lambda x: x[1], reverse=True)

    score = cos_sim_score[1:6]
    tag_indices = [i[0] for i in score]

    return tag_indices


train_data = pd.read_csv(r'train_data_add.csv', encoding="utf-8", low_memory=False)
vectorizer = TfidfVectorizer()
vectorizer.fit(train_data['title'])
title_vectors = vectorizer.transform(train_data['title'])

text = input('타이틀을 입력하세요 : ')
text_vector = vectorizer.transform([text])

top_5_title = train_data['title'].iloc[top5_indices(text_vector, title_vectors, 0)]
print(f"\n{top_5_title}")

# for i in range(len(okky_data)):
#     tit_5_q = okky_data['title'].iloc[top5_indices(text_vector, title_vectors, i)]
#     print(f"{i + 1}번 게실물과 유사한 게시물을 가진 목록\n{tit_5_q}")
