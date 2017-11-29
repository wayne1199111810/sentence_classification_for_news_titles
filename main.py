import json, sys
from include import bag_of_word as BoW
from include import word2vec as w2v
from include.cnn import CNN

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'
BoW_dataset = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'
w2v_train_data = '../data/sentence_classification_for_news_titles/trainCorpora.csv'
w2v_valid_data = '../data/sentence_classification_for_news_titles/validCorpora.csv'
BOW_CONFIG = './config/BoW.json'
W2V_CONFIG = './config/w2v.json'

def preprocessing_pad(x, maxlen):
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x

def run(sentence_representation, model_filename, LOG):
    train_data, test_data, config_file = getDatasetFromSentRepresentation(sentence_representation)
    runCNN(train_data, test_data, config_file, model_filename, LOG)

def getDatasetFromSentRepresentation(sentence_representation):
    if sentence_representation.lower() == 'bow':
        train_x, train_y, test_x, test_y = BoW.getBagOfWords(BoW_dataset)
        config_file = BOW_CONFIG
    elif sentence_representation.lower() == 'w2v':
        train_x, train_y, test_x, test_y = w2v.getWord2Vec(w2v_train_data, w2v_valid_data, pretrain_model)
        config_file = W2V_CONFIG
    return (train_x, train_y), (test_x, test_y), config_file

def runCNN(train_data, test_data, config_file, model_filename, LOG):
    train_x, train_y = train_data[0], train_data[1]
    test_x, test_y = test_data[0], test_data[1]

    cnn = CNN(config_file)
    batch_size, epochs = getEpochAndBatch(config_file)
    hist = cnn.fit(train_x, train_y, batch_size, epochs, model_filename)
    result = cnn.predict(test_x,test_y)
    writeLog(hist, result, LOG)    

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

if __name__ == '__main__':
    sentence_representation = sys.argv[1]
    MODEL_OUTPUT_FILENAME = sys.argv[2]
    LOG = sys.argv[3]
    run(sentence_representation, MODEL_OUTPUT_FILENAME, LOG)