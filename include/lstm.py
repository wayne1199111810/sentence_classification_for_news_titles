import json
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.utils import plot_model

class lstm:
    def __init__(self, config_file):
        self.loadLSTMConfig(config_file)
        self.constructNN()

    def loadLSTMConfig(self, filename):
        with open(filename, 'r') as f:
            para = json.load(f)
            self.model_type = para['model_type']
            self.dropout_p1 = para['dropout_p1']
            self.dropout_p2 = para['dropout_p2']
            self.input_size = (para['dim'], 1)
            self.hidden_notes = para['hidden_notes']
            self.output_size = para['output_size']

    def constructNN(self):
        model_input = Input(shape=self.input_size)
        drop_1 = Dropout(self.dropout_p1)(model_input)
        lstm_layer = LSTM(self.hidden_notes)(drop_1)
        drop_2 = Dropout(self.dropout_p2)(lstm_layer)
        model_output = (Dense(self.output_size, activation='softmax'))(drop_2)
        self.model = Model(model_input, model_output)
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
        predictions = self.model.predict(test_x)
        return s, predictions
