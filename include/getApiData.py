import gensim
import numpy as np
import time
import warnings

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from include.OneHotEncoder import OneHotEncoder

from gensim.models.wrappers import FastText

max_len = 50

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'

# category = set(['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports'])
event_category = set(['Business', 'Games', 'Health', 'Science'])
a3_category = set(['business', 'sport', 'entertainment', 'sci_tech', 'health'])

def countToken(train_x):
    count = 0
    vocab = set()
    for sent in train_x:
        count += len(train_x)
        for w in sent:
            vocab.add(w)
    print('vocab: ' + str(len(vocab)))
    print('tokens: ' + str(count))
    print('utterances: ' + str(len(train_x)))

def pad2MaxLength(words):
    count = 0
    origin_length = len(words)
    while origin_length + count < max_len:
        words.append(words[count%origin_length])
        count += 1

def build_data(file, category):
    features = []
    labels = []
    count_b = 0
    count_s = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens[1:]) > 0:
                if tokens[0] not in category:
                    continue
                labels.append(tokens[0])
                words = tokens[1:]
                pad2MaxLength(words)
                features.append(words)
    return features, labels

def getBowApiData(train_file, valid_file, dataset_category, for_cnn = False,  n_features=1000):
    vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
    train_x, train_y = build_data(train_file, dataset_category)
    valid_x, valid_y = build_data(valid_file, dataset_category)
    train_x = vectorizeBowFeatures(vectorizer, train_x, for_cnn)
    valid_x = vectorizeBowFeatures(vectorizer, valid_x, for_cnn)
    if for_cnn:
        label_dic = list(set(train_y))
        encoder = OneHotEncoder(label_dic)
        train_y = encoder.transform(train_y)
        valid_y = encoder.transform(valid_y)

    return train_x, train_y, valid_x, valid_y

def vectorizeBowFeatures(vectorizer, train_x, for_cnn):
    concatenate_toekns = list()
    for instance in train_x:
        s = ''
        for token in instance:
            s += token + ' '
        concatenate_toekns.append(s.strip())
    features = vectorizer.transform(concatenate_toekns).toarray()
    features = features.reshape(features.shape[0], features.shape[1], 1) if for_cnn else features
    return features

#################################################################################################
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

def getW2vApiData(train_file, valid_file, dataset_category, pretrain_model, for_cnn = False):
    word2vector = build_word2vector(pretrain_model)
    vectorizer = MeanEmbeddingVectorizer(word2vector)
    print('word2vector built')
    train_x, train_y = build_data(train_file, dataset_category)
    label_dic = list(set(train_y))
    print('w2v labels:' + str(len(label_dic)))
    encoder = OneHotEncoder(label_dic)
    train_x, train_y = vectorizew2vFeatures(vectorizer, encoder, train_x, train_y, for_cnn)
    print('training data built')
    valid_x, valid_y = build_data(valid_file, dataset_category)
    valid_x, valid_y = vectorizew2vFeatures(vectorizer, encoder, valid_x, valid_y, for_cnn)
    print('validation data built')
    return train_x, train_y, valid_x, valid_y

def build_word2vector(pretrain_model):
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model, binary=True)
    print('word2vector loaded')
    return dict(zip(model.wv.index2word, model.wv.syn0))

def vectorizew2vFeatures(vectorizer, encoder, x, y, for_cnn):
    x = vectorizer.transform(x)
    if for_cnn:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = encoder.transform(y)
    return x, y
#######################################################################################################
def getFasttextApiData(train_file, valid_file, pretrain_model, for_cnn = False):
    texter = build_fasttexter(pretrain_model)

def build_fasttexter(pretrain_model):
    model = FastText.load_fasttext_format(pretrain_model)
    print('fasttext loaded')
    return model
