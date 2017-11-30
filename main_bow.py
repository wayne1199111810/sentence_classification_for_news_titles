from include.getApiData import getApiData as api
from include import bag_of_word as BoW
from include import io_file
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

UCI_dataset = '../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv'
dataDir = '../data/sentence_classification_for_news_titles/api_data/corpora'
train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora.csv'
valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora.csv'

def getUciData(dataset):
    df = io_file.read_corpora_file(dataset)

    df_train, df_dev, df_test = io_file.split_train_dev_test(df)
    train_all = True
    if train_all:
        data_train = df_train.loc[:, "title"]
        y_train = df_train.loc[0:, "category"]
    else:
        data_train = df_train.loc[:2000, "title"]
        y_train = df_train.loc[0:2000, "category"]
    data_test = df_dev.loc[:, "title"]
    y_test = df_dev.loc[:, "category"]

    vectorizer = HashingVectorizer(stop_words='english')
    x_train = vectorizer.transform(data_train)
    x_test = vectorizer.transform(data_test)

    return x_train, y_train, x_test, y_test

def getP

def runApi():
    api.BOW(train_file, valid_file)

def runUci(dataset = UCI_dataset):
    x_train, y_train, x_test, y_test = getUciData(UCI_dataset)
    # build and train
    classifiers = []

    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))

    for clf, clf_name in classifiers:
        start_time = time.time()
        clf.fit(x_train, y_train)

        # test
        y_test_pred = clf.predict(x_test)
        accuracy = np.sum(y_test_pred == y_test) / len(y_test)
        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))


if __name__ == '__main__':
    data_type = sys.argv[1]
    if data_type == 'api':
        runApi()
    elif data_type == 'UCI':
        runUci()