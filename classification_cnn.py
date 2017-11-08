import numpy as np
import json, sys
from scipy.sparse import csr_matrix

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.datasets import imdb
from bag_of_word import io_file
from bag_of_word import bag_of_word as BoW
from word2vec import word2vec

MAXLEN = 1
BATCH_SIZE = 32
EPOCHS = 10
pretrain_model = './data/GoogleNews-vectors-negative300.bin'
BoW_dataset = './data/newsCorpora.shuffled.csv'
w2v_train_data = './data/trainCorpora.csv'
w2v_valid_data = './data/validCorpora.csv'
BOW_CONFIG = './config/BoW.json'
W2V_CONFIG = './config/w2v.json'

def preprocessing_pad(x, maxlen):
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x

class CNN:
    def __init__(self, config_file):
        self.loadCNNConfig(config_file)
        self.constructNN()

    def loadCNNConfig(self, filename):
        with open(filename, 'r') as f:
            para = json.load(f)
            self.model_type = para['model_type']
            self.dropout_p = para['dropout_p']
            self.windows = para['windows']
            self.num_of_filters = para['num_of_filters']
            # self.input_size = (para['sequence_length'], para['w2v_dim'])
            self.input_size = (para['dim'], 1)
            self.output_size = para['output_size']

    def constructNN(self):
        model_input = Input(shape=self.input_size)
        convs = []
        for window_size in self.windows:
            conv = Convolution1D(filters=self.num_of_filters,
                kernel_size=window_size,
                padding="valid",
                activation="relu",
                strides=1)(model_input)
            max_pool = MaxPooling1D(pool_size=2)(conv)
            max_pool = Flatten()(max_pool)
            convs.append(max_pool)
        merged = Concatenate()(convs) if len(convs) > 1 else convs[0]
        dropout = Dropout(self.dropout_p)(merged)
        # model_ouput = Dense(hidden_dims, activation="relu")(dropout)
        model_ouput = Dense(self.output_size, activation="softmax")(dropout)

        self.model = Model(model_input, model_ouput)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def plot_model(self, filename):
        plot_model(self.model, to_file=filename)

    def fit(self,
        train_x,
        train_y,
        batch_size,
        epochs,
        output_filename,
        valid_x=None,
        valid_y=None):
        if valid_x is not None and valid_y is not None:
            hist = self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
                validation_data=(valid_x, valid_y), verbose=2)
        else:
            hist = self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=2)
        self.model.save(output_filename)
        print(hist.history)
        return hist

    def predict(self, test_x, test_y):
        loss, acc = self.model.evaluate(test_x, test_y)
        s = '\nTesting loss: {}, acc: {}\n'.format(loss, acc)
        print(s)
        return s

def getDatasetFromSentRepresentation(sentence_representation):
    if sentence_representation.lower() == 'bow':
        train_x, test_x, train_y, test_y = BoW.getBagOfWords(BoW_dataset)
        config_file = BOW_CONFIG
    elif sentence_representation.lower() == 'w2v':
        train_x, train_y, test_x, test_y = word2vec.getWord2Vec(w2v_train_data, w2v_valid_data, pretrain_model)
        config_file = W2V_CONFIG
    return (train_x, train_y), (test_x, test_y), config_file

def writeLog(hist, result, LOG):
    f = open(LOG, 'w')
    train_log = hist.history
    f.write(str(train_log) + '\n\n')
    f.write(result)
    f.close()

def getEpochAndBatch(config_file):
    with open(config_file, 'r') as f:
        para = json.load(f)
        return para['batch_size'], para['epochs']

def printDataSize(x, y):
    print(type(x))
    print('x: ' + str(x.shape))
    print('y: ' + str(y.shape))

def runCNN(train_data, test_data, config_file, model_filename, LOG):
    train_x, train_y = train_data[0], train_data[1]
    test_x, test_y = test_data[0], test_data[1]

    cnn = CNN(config_file)
    batch_size, epochs = getEpochAndBatch(config_file)
    hist = cnn.fit(train_x, train_y, batch_size, epochs, model_filename)
    result = cnn.predict(test_x,test_y)
    writeLog(hist, result, LOG)

def run(sentence_representation, model_filename, LOG):
    train_data, test_data, config_file = getDatasetFromSentRepresentation(sentence_representation)
    runCNN(train_data, test_data, config_file, model_filename, LOG)

if __name__ == '__main__':
    sentence_representation = sys.argv[1]
    MODEL_OUTPUT_FILENAME = sys.argv[2]
    LOG = sys.argv[3]
    run(sentence_representation, MODEL_OUTPUT_FILENAME, LOG)


    # train_x, test_x, train_y, test_y = BoW.getBagOfWords(BoW_dataset)
    # train_x = preprocessing_pad(train_x, MAXLEN)
    # cnn = CNN(config_file)
    # cnn.plot_model('cnn')    
    # hist = cnn.fit(train_x, train_y, BATCH_SIZE, EPOCHS, MODEL_OUTPUT_FILENAME)
    # cnn.predict(test_x,test_y)