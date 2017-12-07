import json, sys
from include.lstm import lstm
from include import getApiData as api
from include import getUciData as uci

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'
UCI_BoW = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'
UCI_w2v_train_data = '../data/sentence_classification_for_news_titles/trainCorpora.csv'
UCI_w2v_valid_data = '../data/sentence_classification_for_news_titles/validCorpora.csv'
BOW_CONFIG = './config/lstm_bow.json'
W2V_CONFIG = './config/lstm_w2v.json'

# api_train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora'
# api_valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora'

api_train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_weight'
api_valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_weight'


def preprocessing_pad(x, maxlen):
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x

def run(sentence_representation, model_filename, LOG, dataset):
    if dataset.lower() == 'uci':
        train_data, test_data, config_file = getUciFromSentRepresentation(sentence_representation)
    elif dataset.lower() == 'api':
        train_data, test_data, config_file = getApiFromSentRepresentation(sentence_representation)
    runLSTM(train_data, test_data, config_file, model_filename, LOG)

def getUciFromSentRepresentation(sentence_representation):
    for_cnn = True
    if sentence_representation.lower() == 'bow':
        config_file = BOW_CONFIG
        train_x, train_y, test_x, test_y = uci.getBowUciData(UCI_BoW, for_cnn)
    elif sentence_representation.lower() == 'w2v':
        config_file = W2V_CONFIG
        train_x, train_y, test_x, test_y = uci.getW2vUciData(UCI_w2v_train_data, UCI_w2v_valid_data, pretrain_model, for_cnn)
    return (train_x, train_y), (test_x, test_y), config_file

def getApiFromSentRepresentation(sentence_representation):
    for_cnn = True
    if sentence_representation.lower() == 'bow':
        config_file = BOW_CONFIG
        train_x, train_y, test_x, test_y = api.getBowApiData(api_train_file, api_valid_file, for_cnn)
    elif sentence_representation.lower() == 'w2v':
        config_file = W2V_CONFIG
        train_x, train_y, test_x, test_y = api.getW2vApiData(api_train_file, api_valid_file, pretrain_model, for_cnn)
    return (train_x, train_y), (test_x, test_y), config_file

def runLSTM(train_data, test_data, config_file, model_filename, LOG):
    train_x, train_y = train_data[0], train_data[1]
    test_x, test_y = test_data[0], test_data[1]

    rnn = lstm(config_file)
    batch_size, epochs = getEpochAndBatch(config_file)
    print(model_filename)
    hist = rnn.fit(train_x, train_y, batch_size, epochs, model_filename)
    result, predictions = rnn.predict(test_x,test_y)
    writeLog(hist, result, LOG)
    writeWrong(api_valid_file, predictions, test_y)

def getEpochAndBatch(config_file):
    with open(config_file, 'r') as f:
        para = json.load(f)
        return para['batch_size'], para['epochs']

def writeLog(hist, result, LOG):
    f = open(LOG, 'w')
    train_log = hist.history
    f.write(str(train_log) + '\n\n')
    f.write(result)
    f.close()

def printDataSize(x, y):
    print(type(x))
    print('x: ' + str(x.shape))
    print('y: ' + str(y.shape))

def writeWrong(filename, pred_y, test_y):
    in_f = open(filename, 'r', encoding='utf-8')
    out_f = open('wrong_lstm', 'w', encoding = 'utf-8')
    category = set(['Business', 'Games', 'Health', 'Science'])
    count = 0
    for line in in_f:
        tokens = line.strip().split()
        if len(tokens[1:]) == 0 or tokens[0] not in category:
            continue
        if pred_y[count] != test_y[count]:
            s = str(pred_y[count]) + '::' + line
            out_f.write(s)
        count += 1
    out_f.close()
    in_f.close()

if __name__ == '__main__':
    sentence_representation = sys.argv[1]
    MODEL_OUTPUT_FILENAME = sys.argv[2]
    LOG = sys.argv[3]
    dataset = sys.argv[4]
    run(sentence_representation, MODEL_OUTPUT_FILENAME, LOG, dataset)
