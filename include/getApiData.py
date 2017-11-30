import gensim
import numpy as np
import time

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from include.OneHotEncoder import OneHotEncoder
# from OneHotEncoder import OneHotEncoder


dataDir = '../data/sentence_classification_for_news_titles/api_data/corpora'
pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'

train_file = dataDir + '/trainCorpora.csv'
valid_file = dataDir + '/validCorpora.csv'

category = set(['Business', 'Games', 'Health', 'Science'])
category = set(['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports'])


def build_data(file):
    features = []
    labels = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()[1:-1].split()
            if len(tokens[1:]) > 0:
                if tokens[0] not in category:
                    continue
                labels.append(tokens[0])
                features.append(tokens[1:])
    return features, labels

def getBagOfWords(train_file, valid_file, for_cnn = False,  n_features=1000):
    vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
    train_x, train_y = build_data(train_file)
    valid_x, valid_y = build_data(valid_file)
    train_x = vectorizeBowFeatures(vectorizer, train_x, for_cnn)
    valid_x = vectorizeBowFeatures(vectorizer, valid_x, for_cnn)
    if for_cnn:
        label_dic = list(set(train_y))
        print('BOW labels:' + str(len(label_dic)))
        encoder = OneHotEncoder(label_dic)
        train_y = encoder.transform(train_y)
        valid_y = encoder.transform(valid_y)

    return train_x, train_y, valid_x, valid_y

def vectorizeBowFeatures(vectorizer, raw_features, for_cnn):
    concatenate_toekns = list()
    for instance in raw_features:
        s = ''
        for token in instance:
            s += token + ' '
        concatenate_toekns.append(s.strip())
    features = vectorizer.transform(concatenate_toekns).toarray()
    features = features.reshape(features.shape[0], features.shape[1], 1) if for_cnn else features
    return features

#################################################################################################
def getWord2Vec(train_file, valid_file, pretrain_model, for_cnn = False):
    word2vector = build_word2vector(pretrain_model)
    vectorizer = MeanEmbeddingVectorizer(word2vector)
    print('word2vector built')
    train_x, train_y = build_data(train_file)
    label_dic = list(set(train_y))
    print('w2v labels:' + str(len(label_dic)))
    encoder = OneHotEncoder(label_dic)
    train_x, train_y = vectorizew2vFeatures(vectorizer, encoder, train_x, train_y, for_cnn)
    print('training data built')
    valid_x, valid_y = build_data(valid_file)
    valid_x, valid_y = vectorizew2vFeatures(vectorizer, encoder, valid_x, valid_y, for_cnn)
    print('validation data built')
    return train_x, train_y, valid_x, valid_y

def build_word2vector(pretrain_model):
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model, binary=True)
    print('word2vector loaded')
    return dict(zip(model.wv.index2word, model.wv.syn0))

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

def vectorizew2vFeatures(vectorizer, encoder, x, y, for_cnn):
    x = vectorizer.transform(x)
    if for_cnn:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = encoder.transform(y)
    return x, y

def BOW(train_file=train_file, valid_file=valid_file):
    train_x, train_y, valid_x, valid_y = getBagOfWords(train_file, valid_file)
    run(train_x, train_y, valid_x, valid_y)

def W2V():
    train_x, train_y, valid_x, valid_y = getWord2Vec(train_file, valid_file, pretrain_model)
    run(train_x, train_y, valid_x, valid_y)

def run(train_x, train_y, valid_x, valid_y):
    classifiers = []
    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))
    for clf, clf_name in classifiers:
        start_time = time.time()
        clf.fit(train_x, train_y)
        # test
        y_test_pred = clf.predict(valid_x)
        accuracy = np.sum(y_test_pred == valid_y) / len(valid_y)
        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))

if __name__ == '__main__':
    BOW()
    # W2V()