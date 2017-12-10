import sys
from include.utility import getParaFromConfig
from include.DataProvider import BowDataProvider, W2vDataProvider
from include.cnn import CNN

def runBow(config_file, valid_file):
    para = getParaFromConfig(config_file)
    category = para['category']
    bow = BowDataProvider(category)
    evaluateModel(para, bow, valid_file)

def runW2v(config_file, valid_file, pretrain_model):
    para = getParaFromConfig(config_file)
    category = para['category']
    w2v = W2vDataProvider(category, pretrain_model)
    evaluateModel(para, w2v, valid_file)

def evaluateModel(para, data_provider, valid_file):
    model_name = para['output_model_name']
    cnn = CNN(para)
    cnn.loadModel(model_name)
    print('model loaded')
    valid_x, valid_y = data_provider.getData(valid_file)
    model.predict(valid_x, valid_y)

if __name__ == '__main__':
    sent_repres = sys.argv[1]
    config_file = sys.argv[2]
    valid_file = sys.argv[3]
    if sent_repres.lower() == 'bow':
        runBow(config_file, valid_file)
    elif sent_repres.lower() == 'w2v':
        pretrain_model = sys.argv[4]
        run(config_file, valid_file, pretrain_model)
