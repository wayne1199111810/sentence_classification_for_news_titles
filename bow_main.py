from include import getApiData as api
from include import getUciData as uci
from include import io_file
import sys, time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

UCI_dataset = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'
train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora'
valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora'

def runApi(train_file=train_file, valid_file=valid_file):
    train_x, train_y, test_x, test_y = api.getBowApiData(train_file, valid_file)
    print('data loaded')
    trainBOW(train_x, train_y, test_x, test_y)

def runUci(dataset = UCI_dataset):
    train_x, train_y, test_x, test_y = uci.getBowUciData(UCI_dataset)
    print('data loaded')
    trainBOW(train_x, train_y, test_x, test_y)

def trainBOW(train_x, train_y, test_x, test_y):
    classifiers = []
    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))
    for clf, clf_name in classifiers:

        start_time = time.time()
        clf.fit(train_x, train_y)

        # test
        y_test_pred = clf.predict(test_x)

        # Error_predict_from = dict({'Business':0, 'Games':0, 'Health':0, 'Science':0})
        # error_predict_to = dict({'Business':0, 'Games':0, 'Health':0, 'Science':0})
        # for i in range(len(test_y)):
        #     if test_y[i] != y_test_pred[i]:
        #         Error_predict_from[test_y[i]] += 1
        #         error_predict_to[y_test_pred[i]] += 1
        accuracy = np.sum(y_test_pred == test_y) / len(test_y)

        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))

if __name__ == '__main__':
    print('api')
    runApi()
    print('uci')
    runUci()
