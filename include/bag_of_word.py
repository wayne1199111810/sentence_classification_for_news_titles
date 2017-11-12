import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import time
from include import io_file
from include.OneHotEncoder import OneHotEncoder

label_dic = ['e','t','b','m']
dataset = "../data/newsCorpora.shuffled.csv"

def getBagOfWords(filename, n_features=1000, train_all=True, num_of_instance=100, label_dic=label_dic):
    data_train, data_test, y_train, y_test = splitData2TrainAndTest(filename, train_all, num_of_instance)
    encoder = OneHotEncoder(label_dic)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
    x_train = vectorizeFeatures(vectorizer, data_train, n_features)
    x_test = vectorizeFeatures(vectorizer, data_test, n_features)
    return x_train, y_train, x_test, y_test

def splitData2TrainAndTest(filename, train_all, num_of_instance):
    df = io_file.read_corpora_file(filename)
    df_train, df_dev, df_test = io_file.split_train_dev_test(df)
    if train_all:
        data_train = df_train.loc[:, "title"]
        y_train = df_train.loc[0:, "category"]
    else:
        data_train = df_train.loc[0:num_of_instance, "title"]
        y_train = df_train.loc[0:num_of_instance, "category"]
    data_test = df_dev.loc[:, "title"]
    y_test = df_dev.loc[:, "category"]
    return data_train, data_test, y_train, y_test

def vectorizeFeatures(vectorizer, raw_features, n_features, stop_words='english'):
    features = vectorizer.transform(raw_features).toarray()
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features

if __name__ == "__main__":
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

    # build and train
    classifiers = []

    #classifiers.append((KNeighborsClassifier(n_neighbors=10), "KNN"))
    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))
    #classifiers.append((SVC(C=0.01), "SVM"))

    for clf, clf_name in classifiers:
        start_time = time.time()
        clf.fit(x_train, y_train)

        # test
        x_test = vectorizer.transform(data_test)

        y_test_pred = clf.predict(x_test)
        #print(y_test_pred)

        accuracy = np.sum(y_test_pred == y_test) / len(y_test)

        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))