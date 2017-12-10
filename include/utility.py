import time, json
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

def getParaFromConfig(filename):
    with open(filename, 'r') as f:
        para = json.load(f)
    return para

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
    print('Training Accuracy:', trainer.score(train_x, train_y))
    print('Validation Accuracy:', trainer.score(valid_x, valid_y))
    print("\tPrecision: %1.3f" % precision_score(valid_y, pred_y, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(valid_y, pred_y, average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(valid_y, pred_y, average='weighted'))
    print('Categories: ', trainer.classes_)
