import sys
from include.DataProvider import W2vDataProvider
from include.utility import getParaFromConfig
from include.utility import train_logistic, train_svm

def run(config_file, train_file, test_file, pretrain_model):
    para = getParaFromConfig(config_file)
    category = para['category']
    w2v = W2vDataProvider(category, pretrain_model)
    train_x, train_y = w2v.getData(train_file)
    print('train data built')
    valid_x, valid_y = w2v.getData(valid_file)
    print('valid data built')
    trainW2v(train_x, train_y, valid_x, valid_y)

def trainW2v(train_x, train_y, valid_x, valid_y):
    train_logistic(train_x, train_y, valid_x, valid_y)
    train_svm(train_x, train_y, valid_x, valid_y)

if __name__ == '__main__':
    config_file = sys.argv[1]
    pretrain_model = sys.argv[2]
    train_file = sys.argv[3]
    valid_file = sys.argv[4]

    run(config_file, train_file, valid_file, pretrain_model)
