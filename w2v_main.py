import sys, json
from include.DataProvider import W2vDataProvider

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

pretrain_model = '../data/GoogleNews-vectors-negative300.bin'

def run(config_file, train_file, test_file, pretrain_model):
    para = getParaFrom(config_file)
    category = para['category']
    w2v = W2vDataProvider(category, pretrain_model)
    train_x, train_y = w2v.getData(train_file)
    print('train data built')
    valid_x, valid_y = w2v.getData(valid_file)
    print('valid data built')
    trainW2v(train_x, train_y, valid_x, valid_y)
    del train_x, train_y, valid_x, valid_y

def getParaFrom(filename):
    with open(filename, 'r') as f:
        para = json.load(f)
    return para

def trainW2v(train_x, train_y, valid_x, valid_y):
    train_logistic(train_x, train_y, valid_x, valid_y)
    train_svm(train_x, train_y, valid_x, valid_y)

def train_logistic(train_x, train_y, valid_x, valid_y):
    lr = Pipeline([("logistic regression", LogisticRegression())])
    lr.fit(train_x, train_y)
    print('Logistic Regression trained')
    evaluation(lr, train_x, train_y, valid_x, valid_y)

def train_svm(train_x, train_y, valid_x, valid_y):
    svc = Pipeline([("linear svc", SVC(kernel="linear"))])
    svc.fit(train_x, train_y)
    print('SVM trained')
    evaluation(svc, train_x, train_y, valid_x, valid_y)

def evaluation(trainer, train_x, train_y, valid_x, valid_y):
    pred_y = trainer.predict(valid_x)
    print('SVM Training Accuracy:', trainer.score(train_x, train_y))
    print('SVM Validation Accuracy:', trainer.score(valid_x, valid_y))
    print("\tPrecision: %1.3f" % precision_score(valid_y, pred_y, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(valid_y, pred_y, average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(valid_y, pred_y, average='weighted'))
    print('Categories: ', trainer.classes_)

def writeWrong(pred_y, valid_y, T, outfile, infile):
    in_f = open(infile, 'r', encoding='utf-8')
    out_f = open(outfile, 'w', encoding = 'utf-8')
    category = set(['Business', 'Games', 'Health', 'Science'])
    count = 0
    for line in in_f:
        tokens = line.strip().split()
        if len(tokens[1:]) == 0 or tokens[0] not in category:
            continue
        if pred_y[count] != valid_y[count]:
            s = str(T[count]) + str(pred_y[count]) + '::' + line
            out_f.write(s)
        count += 1
    out_f.close()
    in_f.close()

if __name__ == '__main__':
    config_file = sys.argv[1]
    pretrain_model = sys.argv[2]
    train_file = sys.argv[3]
    valid_file = sys.argv[4]

    run(config_file, train_file, valid_file, pretrain_model)
