# //////////////////////////////Functions//////////////////////////////
#
# metricsResult()
#
# Given the true labels and predicted labels, we compute the precision, recall,
# and f1 score. Return a dict with these three (key, value) pair for the purpose
# of writing to log.
#
#
# getIntFormat()
#
# Since precision_score, recall_score, and f1_score only take binary value instead
# of probability distribution which is produced by CNN softmax output layer. We
# assign 1 to the max index and zeros to the other indices.
#

import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.utils import plot_model
from sklearn.metrics import precision_score, recall_score, f1_score


class CNN:
    def __init__(self, para):
        self.initPara(para)
        self.constructNN()

    def initPara(self, para):
        self.model_type = para['model_type']
        self.dropout_p = para['dropout_p']
        self.windows = para['windows']
        self.num_of_filters = para['num_of_filters']
        self.max_sent_len = para['max_sent_len']
        self.input_size = (self.max_sent_len, para['dim'])
        self.output_size = para['output_size']
        self.batch_size = para['batch_size']
        self.epochs = para['epochs']
        self.output_filename = para['output_model_name']

    def constructNN(self):
        model_input = Input(shape=self.input_size)
        convs = []
        for window_size in self.windows:
            conv = Convolution1D(filters=self.num_of_filters,
                kernel_size=window_size,
                padding="valid",
                activation="relu",
                strides=1)(model_input)
            pool_size = self.max_sent_len - window_size + 1
            max_pool = MaxPooling1D(pool_size=pool_size)(conv)
            max_pool = Flatten()(max_pool)
            convs.append(max_pool)
        merged = Concatenate()(convs) if len(convs) > 1 else convs[0]
        dropout = Dropout(self.dropout_p)(merged)
        model_ouput = Dense(self.output_size, activation="softmax")(dropout)
        self.model = Model(model_input, model_ouput)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        if valid_x is not None and valid_y is not None:
            hist = self.model.fit(train_x, train_y, batch_size=self.batch_size,
             epochs=self.epochs, validation_data=(valid_x, valid_y), verbose=2)
        else:
            hist = self.model.fit(train_x, train_y, batch_size=self.batch_size,
             epochs=self.epochs, verbose=2)
        print(hist.history)
        self.model.save(self.output_filename)
        print('')
        return hist

    def predict(self, test_x, test_y):
        loss, acc = self.model.evaluate(test_x, test_y)
        s = '\nTesting loss: {}, acc: {}\n'.format(loss, acc)
        print(s)
        pred_y = self.model.predict(test_x)
        score = metricsResult(pred_y, test_y)
        score['Test loss'] = loss
        score['Test acc'] = acc
        return pred_y, score

    def loadModel(self, path):
        self.model = load_model(path)

def metricsResult(pred_y, test_y):
    int_test_y = getIntFormat(test_y)
    int_pred_y = getIntFormat(pred_y)
    precision = precision_score(int_test_y, int_pred_y, average='weighted')
    recall = recall_score(int_test_y, int_pred_y, average='weighted')
    f1 = f1_score(int_test_y, int_pred_y, average='weighted')
    print("\tPrecision: %1.3f" % precision)
    print("\tRecall: %1.3f" % recall)
    print("\tF1: %1.3f\n" % f1)
    score = dict()
    score['precision'] = precision
    score['recall'] = recall
    score['f1'] = f1
    return score

def getIntFormat(float_y):
    int_y = np.zeros(float_y.shape)
    idx =  np.argmax(float_y, axis=1)
    for i in range(int_y.shape[0]):
        int_y[i][idx[i]] = 1
    return int_y
