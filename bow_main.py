from include import getApiData as api
from include import getUciData as uci
from include import io_file
import sys, time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

UCI_dataset = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'

a3_train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3'
a3_valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3'

event_train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event'
event_valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event'

event_category = set(['Business', 'Games', 'Health', 'Science'])
a3_category = set(['business', 'sport', 'entertainment', 'sci_tech', 'health'])
uci_category = set(['e', 'b', 't', 'm'])

dataset2category = {
    'a3': a3_category,
    'event': event_category,
    'uci': uci_category
}

def runApi(train_file, valid_file):
    dataset = train_file.strip().split('_')[-1]
    dataset_category = dataset2category[dataset]
    train_x, train_y, test_x, test_y = api.getBowApiData(train_file, valid_file, dataset_category)
    print('data loaded')
    trainBOW(train_x, train_y, test_x, test_y)
    del train_x, train_y, test_x, test_y

def runUci(dataset = UCI_dataset):
    train_x, train_y, test_x, test_y = uci.getBowUciData(UCI_dataset)
    print('data loaded')
    trainBOW(train_x, train_y, test_x, test_y)
    del train_x, train_y, test_x, test_y

def trainBOW(train_x, train_y, test_x, test_y):
    classifiers = []
    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))
    for clf, clf_name in classifiers:

        start_time = time.time()
        clf.fit(train_x, train_y)

        # test
        pred_y = clf.predict(test_x)
        accuracy = np.sum(pred_y == test_y) / len(test_y)
        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))
        print("\tPrecision: %1.3f" % precision_score(test_y, pred_y, average='weighted'))
        print("\tRecall: %1.3f" % recall_score(test_y, pred_y, average='weighted'))
        print("\tF1: %1.3f\n" % f1_score(test_y, pred_y, average='weighted'))
        # T = clf.predict_proba(test_x)
        print(clf.classes_)
        # writeWrong(y_test_pred, test_y, T, 'wrong_bow_' + str(clf_name))

def writeWrong(pred_y, test_y, T, outfile, infile):
    in_f = open(infile, 'r', encoding='utf-8')
    out_f = open(outfile, 'w', encoding = 'utf-8')
    category = set(['Business', 'Games', 'Health', 'Science'])
    count = 0
    for line in in_f:
        tokens = line.strip().split()
        if len(tokens[1:]) == 0 or tokens[0] not in category:
            continue
        if pred_y[count] != test_y[count]:
            s = str(T[count]) + str(pred_y[count]) + '::' + line
            out_f.write(s)
        count += 1
    out_f.close()
    in_f.close()

if __name__ == '__main__':
    data_typy = sys.argv[1]
    train_file = sys.argv[2]
    valid_file = sys.argv[3]
    if data_typy.lower() == 'api':
        print('API')
        runApi(train_file, valid_file)
    elif data_typy.lower() == 'uci':
        print('UCI')
        runUci()
