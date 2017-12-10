# ///////////////////////////MeanEmbeddingVectorizer///////////////////////////
#
# transform()
#
# Using google's w2v, we take the mean of all the word vectors in each sentence
# as the sentence representation for the logistic regression and SVM.
# Return a 2-D array.
#
# /////////////////////////////EmbeddingVectorizer/////////////////////////////
#
# transform()
#
# Using google's w2v, we concatenate the word vectors to the input features.
# Return a 3-D array
#

import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.vectors_by_word = w2v
        # python 2.7
        # self.dim = len(w2v.itervalues().next())
        # python 3.5
        self.dim = len(next(iter(w2v.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([np.mean([self.vectors_by_word[word] if word in self.vectors_by_word else np.zeros(self.dim) for word in list_of_words], axis=0) for list_of_words in X])

class EmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.vectors_by_word = w2v
        # python 2.7
        # self.dim = len(w2v.itervalues().next())
        # python 3.5
        self.dim = len(next(iter(w2v.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([[self.vectors_by_word[word] if word in self.vectors_by_word else np.zeros(self.dim) for word in list_of_words] for list_of_words in X])
