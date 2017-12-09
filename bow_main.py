import sys, time, json
from include.DataProvider import BowDataProvider

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


def run(config_file, train_file, valid_file):
    para = getParaFrom(config_file)
    category = para['category']
    bow = BowDataProvider(category)
    train_x, train_y = bow.getData(train_file)
    valid_x, valid_y = bow.getData(valid_file)
    trainBow(train_x, train_y, valid_x, valid_y)

def getParaFrom(filename):
    with open(filename, 'r') as f:
        para = json.load(f)
    return para

def trainBow(train_x, train_y, valid_x, valid_y):
    train_logistic(train_x, train_y, valid_x, valid_y)
    train_svm(train_x, train_y, valid_x, valid_y)

def train_logistic(train_x, train_y, valid_x, valid_y):
    trainer_name = 'logistic regression'
    lr = Pipeline([("logistic regression", LogisticRegression())])
    start_time = time.time()
    lr.fit(train_x, train_y)
    print('Logistic Regression trained')
    print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, trainer_name))
    evaluation(lr, train_x, train_y, valid_x, valid_y)

def train_svm(train_x, train_y, valid_x, valid_y):
    trainer_name = 'linear svc'
    svc = Pipeline([("linear svc", SVC(kernel="linear"))])
    start_time = time.time()
    svc.fit(train_x, train_y)
    print('SVM trained')
    print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, trainer_name))
    evaluation(svc, train_x, train_y, valid_x, valid_y)

def evaluation(trainer, train_x, train_y, valid_x, valid_y):
    pred_y = trainer.predict(valid_x)
    print('SVM Training Accuracy:', trainer.score(train_x, train_y))
    print('SVM Validation Accuracy:', trainer.score(valid_x, valid_y))
    print("\tPrecision: %1.3f" % precision_score(valid_y, pred_y, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(valid_y, pred_y, average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(valid_y, pred_y, average='weighted'))
    print('Categories: ', trainer.classes_)

if __name__ == '__main__':
    config_file = sys.argv[1]
    train_file = sys.argv[2]
    valid_file = sys.argv[3]
    run(config_file, train_file, valid_file)
