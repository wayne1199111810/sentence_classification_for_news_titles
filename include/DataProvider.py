# //////////////////////////////Functions//////////////////////////////
#
# printDatasetInfo()
#
# Print the number of unique vocabularies in the dataset, number of words in
# the dataset, and the number of news title in the dataset
#
#
# build_data()
#
# Given the dataset with the first word as the label, and the remaining words as
# the news title. Since we only consider the result for the given category, we
# skip the instances which are not in the category
#
# //////////////////////////////BowDataProvider//////////////////////////////
#
# vectorizeBowFeatures()
#
# Since cnn takes 1*(# of categories) vector as the label, we use one hot as the
# encoding for the categories. Furthermore, cnn takes 3-D array as input, we
# reshape the bow words to 3-D array.
#
# //////////////////////////////W2vDataProvider//////////////////////////////
#
# build_word2vector()
#
# We use google's pretrain w2v model as our word embedding
#
#
# vectorizew2vFeatures()
#
# Since cnn takes 1*(# of categories) vector as the label, we use one hot as the
# encoding for the categories. Furthermore, cnn takes 3-D array as input, we
# reshape the bow words to 3-D array. In order to have a fix size input for CNN,
# we take the maximum sentence length from the dataset and pad zeros to the
# sentences whose length are shorter than maximum length
#

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from include.OneHotEncoder import OneHotEncoder
from include.Vectorizer import MeanEmbeddingVectorizer, EmbeddingVectorizer


def printDatasetInfo(features):
    count = 0
    vocab = set()
    for sent in features:
        count += len(features)
        for w in sent:
            vocab.add(w)
    print('vocab: ' + str(len(vocab)))
    print('tokens: ' + str(count))
    print('utterances: ' + str(len(features)))

def build_data(file, category):
    features = []
    labels = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens[1:]) > 0:
                if tokens[0] not in category:
                    continue
                labels.append(tokens[0])
                features.append(tokens[1:])
    return features, labels

class BowDataProvider(object):
    def __init__(self, category, for_cnn=False, n_features=1000):
        self.category = category
        self.for_cnn = for_cnn
        self.vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
        # self.vectorizer = HashingVectorizer(stop_words='english')
        self.encoder = OneHotEncoder(list(category))

    def getData(self, filename):
        x, y = build_data(filename, self.category)
        return self.vectorizeBowFeatures(x, y)

    def vectorizeBowFeatures(self, x, y):
        concatenate_toekns = list()
        for instance in x:
            s = ''
            for token in instance:
                s += token + ' '
            concatenate_toekns.append(s.strip())
        x = self.vectorizer.transform(concatenate_toekns)
        if self.for_cnn:
            x = x.toarray()
            x = x.reshape(x.shape[0], x.shape[1], 1)
            y = self.encoder.transform(y)
        return x, y

class W2vDataProvider(object):
    def __init__(self, category, pretrain_model=None, for_cnn=False, max_len=100):
        self.category = category
        self.for_cnn = for_cnn
        self.max_len = max_len
        self.encoder = OneHotEncoder(list(category))
        word2vector = self.build_word2vector(pretrain_model)
        self.vectorizer = EmbeddingVectorizer(word2vector) if for_cnn else MeanEmbeddingVectorizer(word2vector)
        print('word2vector built')

    def build_word2vector(self, pretrain_model):
        model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model, binary=True)
        print('word2vector loaded')
        return dict(zip(model.wv.index2word, model.wv.syn0))

    def getData(self, filename):
        x, y = build_data(filename, self.category)
        return self.vectorizew2vFeatures(x, y)

    def vectorizew2vFeatures(self, x, y):
        x = self.vectorizer.transform(x)
        if self.for_cnn:
            tmp = np.zeros((x.shape[0], self.max_len, self.vectorizer.dim))
            for i in range(x.shape[0]):
                if len(x[i]) <= self.max_len:
                    tmp[i][0:len(x[i]),:] = np.array(x[i])
                else:
                    tmp[i][0:self.max_len,:] = np.array(x[i])[:self.max_len][:]
            x = tmp
            y = self.encoder.transform(y)
        return x, y
