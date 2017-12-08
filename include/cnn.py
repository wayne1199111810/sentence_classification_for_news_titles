import json
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.utils import plot_model
from sklearn.metrics import precision_score, recall_score, f1_score

from keras import backend as K

 def mcor(y_true, y_pred):
     #matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

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
            self.max_sent_len = para['max_sent_len']
            self.input_size = (self.max_sent_len, para['dim'])
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
            # pool_size = 2
            pool_size = self.max_sent_len - window_size + 1
            max_pool = MaxPooling1D(pool_size=pool_size)(conv)
            max_pool = Flatten()(max_pool)
            convs.append(max_pool)
        merged = Concatenate()(convs) if len(convs) > 1 else convs[0]
        dropout = Dropout(self.dropout_p)(merged)
        # model_ouput = Dense(hidden_dims, activation="relu")(dropout)
        model_ouput = Dense(self.output_size, activation="softmax")(dropout)

        self.model = Model(model_input, model_ouput)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", recall, precision, f1])

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
