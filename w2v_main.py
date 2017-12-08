import sys

from include import getApiData as api
from include import getUciData as uci

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

pretrain_model = '../data/sentence_classification_for_news_titles/GoogleNews-vectors-negative300.bin'
UCI_train_data = '../data/sentence_classification_for_news_titles/trainCorpora.csv'
UCI_valid_data = '../data/sentence_classification_for_news_titles/validCorpora.csv'

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

def runApi(train_file, test_file, pretrain_model):
    dataset = train_file.strip().split('_')[-1]
    dataset_category = dataset2category[dataset]
    train_x, train_y, test_x, test_y = api.getW2vApiData(train_file, test_file, dataset_category, pretrain_model)
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
    pred_y = lr.predict(test_x)
    print('Logistic Regression trained')
    print('Logistic Regression Training Accuracy:', lr.score(train_x, train_y))
    print('Logistic Regression Validation Accuracy:', lr.score(test_x, test_y))
    print("\tPrecision: %1.3f" % precision_score(test_y, pred_y, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(test_y, pred_y, average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(test_y, pred_y, average='weighted'))
    print(lr.classes_)
    # T = lr.predict_proba(test_x)
    # writeWrong(lr.predict(test_x), test_y, T, 'wrong_w2v_logist')

def train_svm(train_x, train_y, test_x, test_y):
    svc = Pipeline([("linear svc", SVC(kernel="linear"))])
    svc.fit(train_x, train_y)
    pred_y = svc.predict(test_x)
    print('SVM trained')
    print('SVM Training Accuracy:', svc.score(train_x, train_y))
    print('SVM Validation Accuracy:', svc.score(test_x, test_y))
    print("\tPrecision: %1.3f" % precision_score(test_y, pred_y, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(test_y, pred_y, average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(test_y, pred_y, average='weighted'))
    print(svc.classes_)
    # T = svc.predict_proba(test_x)
    # writeWrong(svc.predict(test_x), test_y, T, 'wrong_w2v_svm')

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
    print(train_file)
    if data_typy.lower() == 'api':
        print('API')
        runApi(train_file, valid_file, pretrain_model)
    elif data_typy.lower() == 'uci':
        print('UCI')
        runUCI(train_file, valid_file, pretrain_model)
