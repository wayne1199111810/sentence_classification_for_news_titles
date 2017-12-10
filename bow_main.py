import sys
from include.DataProvider import BowDataProvider
from include.utility import getParaFromConfig
from include.utility import train_logistic, train_svm

def run(config_file, train_file, valid_file):
    para = getParaFromConfig(config_file)
    category = para['category']
    bow = BowDataProvider(category)
    train_x, train_y = bow.getData(train_file)
    valid_x, valid_y = bow.getData(valid_file)
    trainBow(train_x, train_y, valid_x, valid_y)

def trainBow(train_x, train_y, valid_x, valid_y):
    train_logistic(train_x, train_y, valid_x, valid_y)
    train_svm(train_x, train_y, valid_x, valid_y)

if __name__ == '__main__':
    config_file = sys.argv[1]
    train_file = sys.argv[2]
    valid_file = sys.argv[3]
    run(config_file, train_file, valid_file)
