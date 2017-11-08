import sys
import numpy as np
import gensim
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from OneHotEncoder import OneHotEncoder

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

label_encodings = {
    'b': 0,
    't': 1,
    'e': 2,
    'm': 3
};

pretrain_model = '../data/GoogleNews-vectors-negative300.bin'

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

def run(train_file, val_file):
    word2vector = build_word2vector()
    print('word2vector built')
    trainX, trainY = build_data(train_file)
    print('training data built')
    valX, valY = build_data(val_file)
    print('validation data built')
    # train_logistic(word2vector, trainX, trainY, valX, valY)
    train_svm(word2vector, trainX, trainY, valX, valY)

def build_word2vector(pretrain_model):
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model, binary=True)
    print('word2vector loaded')
    return dict(zip(model.wv.index2word, model.wv.syn0))

def build_data(file):
    features = []
    labels = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            features.append(tokens[1].split(' '))
            labels.append(tokens[-4])
    return features, [encode_label(label) for label in labels]

def encode_label(label):
    return label_encodings[label]

def train_logistic(word2vector, trainX, trainY, valX, valY):
    lr = Pipeline([("mean_vectorizer", MeanEmbeddingVectorizer(word2vector)), ("logistic regression", LogisticRegression())])
    lr.fit(trainX, trainY)
    print('Logistic Regression trained')
    print('Logistic Regression Training Accuracy:', lr.score(trainX, trainY))
    print('Logistic Regression Validation Accuracy:', lr.score(valX, valY))

def train_svm(word2vector, trainX, trainY, valX, valY):
    svc = Pipeline([("mean_vectorizer", MeanEmbeddingVectorizer(word2vector)), ("linear svc", SVC(kernel="linear"))])
    svc.fit(trainX, trainY)
    print('SVM trained')
    print('SVM Training Accuracy:', svc.score(trainX, trainY))
    print('SVM Validation Accuracy:', svc.score(valX, valY))

def vectorizeFeatures(vectorizer, encoder, x, y):
    x = vectorizer.transform(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = encoder.transform(y)
    return x, y

def getWord2Vec(train_file, val_file, pretrain_model):
    word2vector = build_word2vector(pretrain_model)
    vectorizer = MeanEmbeddingVectorizer(word2vector)
    encoder = OneHotEncoder(list(label_encodings.values()))
    print('word2vector built')
    trainX, trainY = build_data(train_file)
    trainX, trainY = vectorizeFeatures(vectorizer, encoder, trainX, trainY)
    print('training data built')
    valX, valY = build_data(val_file)
    valX, valY = vectorizeFeatures(vectorizer, encoder, valX, valY)
    print('validation data built')
    return trainX, trainY, valX, valY

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
