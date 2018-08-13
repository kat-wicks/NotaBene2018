'''
for tf-idf cosine similarity and fasttext
'''
import pandas as pd
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import itertools

dataframe = pd.read_csv('C:/Users/ktwic/Desktop/lol.csv')

rows = np.concatenate((dataframe["body"].values.astype('U'), dataframe["marked_text"].values.astype('U'), dataframe["marked_par"].values.astype('U')), axis =0)
stop = set(stopwords.words('english'))
stop= stop.union({'.',',','.','\"', '\'', '?', '!'})
ps = PorterStemmer()

sent_list = [[ps.stem(w.lower()) for w in word_tokenize(i) if w not in stop] for i in rows]

ft_model= FastText()
ft_model.build_vocab(sent_list, min_count =2)
ft_model.train(sent_list, total_examples=ft_model.corpus_count, epochs=ft_model.iter)


comments = [[ps.stem(w.lower()) for w in word_tokenize(i) if w not in stop] for i in dataframe["body"].values.astype('U')]
text = [[ps.stem(w.lower()) for w in word_tokenize(i) if w not in stop] for i in dataframe["marked_text"].values.astype('U')]
par = [[ps.stem(w.lower()) for w in word_tokenize(i) if w not in stop] for i in dataframe["marked_par"].values.astype('U')]

text_similarity = np.zeros(shape = (len(comments)))
paragraph_similarity= np.zeros(shape = (len(comments)))

for c in range(len(comments)):
    comment_vec = np.zeros(shape = (len(comments[c]), 100))
    for word in range(len(comments[c])):
        try:
            comment_vec[word] = ft_model[comments[c][word]]
        except:
            pass
            #print(comments[c][word], " not in model")
    comment_vec = comment_vec.mean(axis=0)
    text_vec = np.zeros(shape = (len(text[c]),100))
    for word in range(len(text[c])):
        try:
            text_vec[word] =(ft_model[text[c][word]])
        except:
            pass
            #print(text[c][word], " not in model")
    text_vec = text_vec.mean(axis=0)
    par_vec = np.zeros(shape = (len(par[c]),100))
    for word in range(len(par[c])):
        try:
            par_vec[word] =(ft_model[par[c][word]])
        except:
            pass
    par_vec = par_vec.mean(axis=0)
    text_similarity[c] = np.dot(comment_vec, text_vec)/(np.linalg.norm(comment_vec)* np.linalg.norm(text_vec))
    paragraph_similarity[c] = np.dot(comment_vec, par_vec)/(np.linalg.norm(comment_vec)* np.linalg.norm(par_vec))


def process(sent):
    s = ' '.join([ps.stem(w.lower()) for w in word_tokenize(sent) if w not in stop])
    return s

tf_idf_c = TfidfVectorizer(stop_words = "english", strip_accents = "unicode")
rows = np.concatenate((dataframe["body"].values.astype('U'), dataframe["marked_text"].values.astype('U'), dataframe["marked_par"].values.astype('U')), axis =0)
tfidf = tf_idf_c.fit_transform(rows)
dataframe["text_sim"] = cosine_similarity(tfidf[0:dataframe.shape[0]], tfidf[dataframe.shape[0]:dataframe.shape[0]*2]).diagonal()
dataframe["para_sim"] = cosine_similarity(tfidf[0:dataframe.shape[0]], tfidf[dataframe.shape[0]*2:dataframe.shape[0]*3]).diagonal()
dataframe["fast_text"] = text_similarity
dataframe["fast_par"] = paragraph_similarity

dataframe.to_csv("lol.csv")