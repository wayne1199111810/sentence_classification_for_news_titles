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

def run(valid_file, config_file, model):
    dataset = valid_file.strip().split('_')[-1]
    category = dataset2category[dataset]
    cnn = CNN(config_file)
    cnn.loadModel(model)
    print('CNN loaded')
    train_data, test_data = getApiFromSentRepresentation(valid_file, valid_file, category)
    test_x, test_y = test_data[0], test_data[1]
    result, predictions = cnn.predict(test_x,test_y)

def getApiFromSentRepresentation(train_file, valid_file, category):
    for_cnn = True
    train_x, train_y, test_x, test_y = api.getBowApiData(valid_file, valid_file, category, for_cnn)
    # train_x, train_y, test_x, test_y = api.getW2vApiData(train_file, valid_file, category, pretrain_model, for_cnn)
    return (train_x, train_y), (test_x, test_y)

# def runCNN(test_data, config_file):

if __name__ == '__main__':
    valid_file = sys.argv[1]
    config_file = sys.argv[2]
    model = sys.argv[3]
    print('API')
    run(valid_file, config_file, model)
