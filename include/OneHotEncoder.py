from sklearn import preprocessing

class OneHotEncoder:
    def __init__(self, dic):
        self.encoder = preprocessing.LabelEncoder()
        self.onehot = preprocessing.OneHotEncoder()
        self.fit(dic)

    def fit(self, dic):
        self.encoder.fit(dic)
        encode_words = self.encoder.transform(dic)
        self.onehot.fit([[w] for w in encode_words])

    def transform(self, input_vector):
        encode_labels = list([[c] for c in self.encoder.transform(input_vector)])
        return self.onehot.transform(encode_labels).toarray()