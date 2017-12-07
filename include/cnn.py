import json
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.utils import plot_model

from sklearn.metrics import precision_score, recall_score, f1_score


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
            self.max_len = para['max_len']

    def constructNN(self):
        model_input = Input(shape=self.input_size)
        convs = []
        for window_size in self.windows:
            conv = Convolution1D(filters=self.num_of_filters,
                kernel_size=window_size,
                padding="valid",
                activation="relu",
                strides=1)(model_input)
            max_pool = MaxPooling1D(pool_size=self.max_len)(conv)
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
        pred_y = self.model.predict(test_x)
        return s, pred_y
