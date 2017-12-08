import json, sys
from include.cnn import CNN
from include import getApiData as api
from include import getUciData as uci

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'
UCI_BoW = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'
UCI_w2v_train_data = '../data/sentence_classification_for_news_titles/trainCorpora.csv'
UCI_w2v_valid_data = '../data/sentence_classification_for_news_titles/validCorpora.csv'

event_category = set(['Business', 'Games', 'Health', 'Science'])
a3_category = set(['business', 'sport', 'entertainment', 'sci_tech', 'health'])
uci_category = set(['e', 'b', 't', 'm'])

dataset2category = {
    'a3': a3_category,
    'event': event_category,
    'uci': uci_category
}

def runApi(train_file, valid_file, config_file, sent_repres, LOG, model_filename):
    dataset = train_file.strip().split('_')[-1]
    category = dataset2category[dataset]
    train_data, test_data = getApiFromSentRepresentation(train_file, valid_file, sent_repres, category)
    runCNN(train_data, test_data, config_file, model_filename, LOG)

def getApiFromSentRepresentation(train_file, valid_file, sent_repres, category):
    for_cnn = True
    if sent_repres.lower() == 'bow':
        train_x, train_y, test_x, test_y = api.getBowApiData(train_file, valid_file, category, for_cnn)
    elif sent_repres.lower() == 'w2v':
        train_x, train_y, test_x, test_y = api.getW2vApiData(train_file, valid_file, category, pretrain_model, for_cnn)
    return (train_x, train_y), (test_x, test_y)

def runUci(train_file, valid_file, config_file, sent_repres, LOG, model_filename):
    train_data, test_data = getUciFromSentRepresentation(train_file, valid_file, sent_repres)
    runCNN(train_data, test_data, config_file, model_filename, LOG)

def getUciFromSentRepresentation(train_file, valid_file, sent_repres):
    for_cnn = True
    if sent_repres.lower() == 'bow':
        train_x, train_y, test_x, test_y = uci.getBowUciData(train_file, for_cnn)
    elif sent_repres.lower() == 'w2v':
        train_x, train_y, test_x, test_y = uci.getW2vUciData(train_file, valid_file, pretrain_model, for_cnn)
    return (train_x, train_y), (test_x, test_y)

def runCNN(train_data, test_data, config_file, model_filename, LOG):
    train_x, train_y = train_data[0], train_data[1]
    test_x, test_y = test_data[0], test_data[1]

    cnn = CNN(config_file)
    batch_size, epochs = getEpochAndBatch(config_file)
    hist = cnn.fit(train_x, train_y, batch_size, epochs, model_filename)
    result, predictions = cnn.predict(test_x,test_y)
    writeLog(hist, result, LOG)
    # writeWrong(api_valid_file, predictions, test_y)

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
    out_f = open('wrong_cnn', 'w', encoding = 'utf-8')
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
    data_type = sys.argv[1]
    sent_repres = sys.argv[2]
    train_file = sys.argv[3]
    valid_file = sys.argv[4]
    config_file = sys.argv[5]
    LOG = sys.argv[6]
    MODEL_OUTPUT_FILENAME = sys.argv[7]

    if data_type.lower() == 'api':
        print('API')
        runApi(train_file, valid_file, config_file, sent_repres, LOG, MODEL_OUTPUT_FILENAME)
    elif data_type == 'uci':
        print('UCI')
        runUci(train_file, valid_file, config_file, sent_repres, LOG, MODEL_OUTPUT_FILENAME)
