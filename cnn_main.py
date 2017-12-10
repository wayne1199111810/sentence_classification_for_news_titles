import json, sys
from include.cnn import CNN
from include.DataProvider import BowDataProvider, W2vDataProvider
from include.utility import getParaFromConfig

def runBow(config_file, train_file, valid_file):
    para = getParaFromConfig(config_file)
    category = para['category']
    provider = BowDataProvider(category, True)
    runCNN(para, provider, train_file, valid_file)

def runW2v(config_file, train_file, valid_file, pretrain_model):
    para = getParaFromConfig(config_file)
    category = para['category']
    max_len = para['max_sent_len']
    provider = W2vDataProvider(category, pretrain_model, True, max_len)
    runCNN(para, provider, train_file, valid_file)

def runCNN(para, provider, train_file, valid_file):
    train_x, train_y = provider.getData(train_file)
    valid_x, valid_y = provider.getData(valid_file)
    cnn = CNN(para)
    hist = cnn.fit(train_x, train_y)
    pred_y, score = cnn.predict(valid_x, valid_y)

    log = {**hist.history, **score}
    writeLog(log, para['log_file'])

def writeLog(log, log_file):
    f = open(log_file, 'w')
    f.write(str(log))
    f.close()

if __name__ == '__main__':
    sent_repres = sys.argv[1]
    config_file = sys.argv[2]
    train_file = sys.argv[3]
    valid_file = sys.argv[4]

    if sent_repres.lower() == 'bow':
        runBow(config_file, train_file, valid_file)
    elif sent_repres.lower() == 'w2v':
        pretrain_model = sys.argv[5]
        runW2v(config_file, train_file, valid_file, pretrain_model)
