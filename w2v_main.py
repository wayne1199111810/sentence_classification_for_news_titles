from include import getApiData as api
from include import getUciData as uci

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'
UCI_train_data = '../data/sentence_classification_for_news_titles/trainCorpora.csv'
UCI_valid_data = '../data/sentence_classification_for_news_titles/validCorpora.csv'

api_train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora'
api_valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora'

def runApi(train_file, test_file, pretrain_model):
    train_x, train_y, test_x, test_y = api.getW2vApiData(train_file, test_file, pretrain_model)
    trainW2v(train_x, train_y, test_x, test_y)
    del train_x, train_y, test_x, test_y

def runUCI(train_file, test_file, pretrain_model):
    train_x, train_y, test_x, test_y = uci.getW2vUciData(train_file, test_file, pretrain_model)
    trainW2v(train_x, train_y, test_x, test_y)
    del train_x, train_y, test_x, test_y

def trainW2v(train_x, train_y, test_x, test_y):
    train_logistic(train_x, train_y, test_x, test_y)
    train_svm(train_x, train_y, test_x, test_y)

def train_logistic(train_x, train_y, test_x, test_y):
    lr = Pipeline([("logistic regression", LogisticRegression())])
    lr.fit(train_x, train_y)
    print('Logistic Regression trained')
    print('Logistic Regression Training Accuracy:', lr.score(train_x, train_y))
    print('Logistic Regression Validation Accuracy:', lr.score(test_x, test_y))

def train_svm(train_x, train_y, test_x, test_y):
    svc = Pipeline([("linear svc", SVC(kernel="linear"))])
    svc.fit(train_x, train_y)
    print('SVM trained')
    print('SVM Training Accuracy:', svc.score(train_x, train_y))
    print('SVM Validation Accuracy:', svc.score(test_x, test_y))

if __name__ == '__main__':
    print('api')
    runApi(api_train_file, api_valid_file, pretrain_model)
    print('UCI')
    runUCI(UCI_train_data, UCI_valid_data, pretrain_model)
