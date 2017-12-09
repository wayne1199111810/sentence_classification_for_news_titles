import json, sys
from include.cnn import CNN
from include.DataProvider import BowDataProvider, W2vDataProvider

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'

def getParaFrom(filename):
    with open(filename, 'r') as f:
        para = json.load(f)
    return para

def runBow(config_file, train_file, valid_file):
    para = getParaFrom(config_file)
    category = para['category']
    provider = BowDataProvider(category, True)
    runCNN()

def runW2v(config_file, train_file, valid_file, pretrain_model):
    para = getParaFrom(config_file)
    category = para['category']
    max_len = para['max_len']
    provider = W2vDataProvider(category, pretrain_model, True, max_len)
    runCNN()

def runCNN(para, provider, train_file, valid_file):
    train_x, train_y = provider.getData(train_file)
    valid_x, valid_y = provider.getData(valid_file)
    cnn = CNN(para)
    hist = cnn.fit(train_x, train_y)
    result, predictions = cnn.predict(test_x,test_y)
    writeLog(hist, result, LOG)

def runApi(train_file, valid_file, config_file, sent_repres, LOG, model_filename):
    train_data, test_data = getApiFromSentRepresentation(train_file, valid_file, sent_repres, category)
    runCNN(train_data, test_data, config_file, model_filename, LOG)

def getApiFromSentRepresentation(train_file, valid_file, sent_repres, category):
    for_cnn = True
    if sent_repres.lower() == 'bow':
        train_x, train_y, test_x, test_y = api.getBowApiData(train_file, valid_file, category, for_cnn)
    elif sent_repres.lower() == 'w2v':
        train_x, train_y, test_x, test_y = api.getW2vApiData(train_file, valid_file, category, pretrain_model, for_cnn)
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

# def writeWrong(filename, pred_y, test_y):
#     in_f = open(filename, 'r', encoding='utf-8')
#     out_f = open('wrong_cnn', 'w', encoding = 'utf-8')
#     category = set(['Business', 'Games', 'Health', 'Science'])
#     count = 0
#     for line in in_f:
#         tokens = line.strip().split()
#         if len(tokens[1:]) == 0 or tokens[0] not in category:
#             continue
#         if pred_y[count] != test_y[count]:
#             s = str(pred_y[count]) + '::' + line
#             out_f.write(s)
#         count += 1
#     out_f.close()
#     in_f.close()

if __name__ == '__main__':
    sent_repres = sys.argv[2]
    config_file = sys.argv[5]
    train_file = sys.argv[3]
    valid_file = sys.argv[4]
    pretrain_model = sys.argv[5]

    if sent_repres.lower() == 'bow':
        runBow(config_file, train_file, valid_file)
    elif sent_repres.lower() == 'w2v':
        runW2v(config_file, train_file, valid_file, pretrain_model)
