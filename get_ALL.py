import sklearn
import nltk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pickle

path = '/home/rohan/eclipse-workspace/ArrayFire_Tests/dataset/'
target = '/home/rohan/eclipse-workspace/ArrayFire_Tests/'
count_vectorizer_pickle = '/home/rohan/eclipse-workspace/ArrayFire_Tests/build/CountsVectorizer.pickle'
tfidf_pickle = '/home/rohan/eclipse-workspace/ArrayFire_Tests/build/TfidfTransformer.pickle'
with open(count_vectorizer_pickle, 'rb') as f:
    movie_vec = pickle.load(f)
with open(tfidf_pickle, 'rb') as f:
    tfidf_transformer_mov = pickle.load(f)
movietrain = load_files(path, shuffle=True)
#movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.transform(movietrain.data)
#tfidf_transformer_mov = TfidfTransformer()
movie_tfidf = tfidf_transformer_mov.transform(movie_counts)
arrX = movie_tfidf.toarray()
arrY = movietrain.target
numrows = arrX.shape[0]
numcols = arrX.shape[1]

assert numrows == arrY.shape[0]
# for i in range(total//200):
#     print(i*200, (i+1)*200)
np.savetxt(target + 'X_tftdf_{}_{}_{}.csv'.format(numrows, numcols, 'ALL'), arrX, fmt='%2.10f', delimiter=' ', newline='\n')
np.savetxt(target + 'Y_{}_{}.csv'.format(numrows, 'ALL'), arrY, fmt='%2.10f', delimiter=' ', newline='\n')
