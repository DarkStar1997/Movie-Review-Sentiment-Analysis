import pickle
import sklearn
#import nltk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
def file2TFIDF():
    with open('./test.txt', 'rb') as f:
        string = f.read()
    with open('./CountsVectorizer.pickle', 'rb') as handle:
        count_vec = pickle.load(handle)
    with open('./TfidfTransformer.pickle', 'rb') as handle:
        tfidf_transf = pickle.load(handle)
    outp = tfidf_transf.transform(count_vec.transform([string])).toarray()
    for i in range(outp.shape[0]):
        for j in range(outp.shape[1]):
            print(str(outp[i,j]))
#    print(string)

file2TFIDF()
