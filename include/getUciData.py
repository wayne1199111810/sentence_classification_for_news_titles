import gensim
import numpy as np
import warnings

from sklearn.feature_extraction.text import HashingVectorizer
from include.OneHotEncoder import OneHotEncoder
from include import io_file

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
label_encodings = {
    'b': 0,
    't': 1,
    'e': 2,
    'm': 3
};
label_dic = ['e','t','b','m']

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

def getBowUciData(filename, for_cnn=False, n_features=1000, label_dic=label_dic):
    train_x, train_y, test_x, test_y = splitData2TrainAndTest(filename)

    encoder = OneHotEncoder(label_dic)
    if for_cnn:
        vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
    else:
        vectorizer = HashingVectorizer(stop_words='english')
    train_x, train_y = vectorizeFeatures(vectorizer, encoder, train_x, train_y, for_cnn)
    test_x, test_y = vectorizeFeatures(vectorizer, encoder, test_x, test_y, for_cnn)
    return train_x, train_y, test_x, test_y

def splitData2TrainAndTest(filename):
    df = io_file.read_corpora_file(filename)
    df_train, df_dev, df_test = io_file.split_train_dev_test(df)
    train_x = df_train.loc[:, "title"]
    train_y = df_train.loc[:, "category"]
    test_x = df_dev.loc[:, "title"]
    test_y = df_dev.loc[:, "category"]
    return train_x, train_y, test_x, test_y

def vectorizeFeatures(vectorizer, encoder, x, y, for_cnn):
    x = vectorizer.transform(x)
    # x = x.toarray() if type(x) != type(np.array([])) else x
    if for_cnn:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = encoder.transform(y)
    return x, y

def vectorizeBowFeatures(vectorizer, encoder, x, y, for_cnn):
    x = vectorizer.transform(x).toarray()
    if for_cnn:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = encoder.transform(y)
    return x, y

def vectorizew2vFeatures(vectorizer, encoder, x, y, for_cnn):
    x = vectorizer.transform(x)
    if for_cnn:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = encoder.transform(y)
    return x, y

##########################################################################
def getW2vUciData(train_file, val_file, pretrain_model, for_cnn=False):
    train_x, train_y = build_data(train_file)
    countToken(train_x)

    word2vector = build_word2vector(pretrain_model)
    vectorizer = MeanEmbeddingVectorizer(word2vector)
    encoder = OneHotEncoder(list(label_encodings.values()))
    print('word2vector built')
    train_x, train_y = build_data(train_file)

    train_x, train_y = vectorizeFeatures(vectorizer, encoder, train_x, train_y, for_cnn)
    print('training data built')
    valid_x, valid_y = build_data(val_file)
    valid_x, valid_y = vectorizeFeatures(vectorizer, encoder, valid_x, valid_y, for_cnn)
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
