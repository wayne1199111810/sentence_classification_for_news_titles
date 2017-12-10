# Sentence Classification for News Titles

### Table of Content
**[Project Description](#project-description)**<br>
**[Dataset](#dataset)**<br>
**[Pre-request](#dependency)**<br>
**[Training Example](#implementation)**<br>

## Project Description
Classifying the semantic content is one of the critical problems in natural language processing. There are many cases where only a small number of words are provided to interpret the meaning or intent such as keyword searches. However, the performance of short text classification is limited due to shortness of sentences, which causes sparse vector representations if we use word occurrence to represent sentences, and lack of context. On the other hand, news titles, though consisting of short sentences, provide rich information of the semantic content in a concise way. Because of this property, we believe that news title classification will be a good start point for our sentence classification task.

## Dataset
### [News Aggregator Data Set](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)
Reference to news web pages collected from an online aggregator in the period from March 10 to August 10 of 2014. The resources are grouped into clusters that represent pages discussing the same news story. The dataset includes also references to web pages that point (has a link to) one of the news page in the collection.

### [Event Registry](http://eventregistry.org/documentation?tab=searchArticles)
It is collected from Event Registry API, in which contains much more up-to-date news. The downside is that they are using their own scraper or classifier, which may introduce more noise to the data compared to manually labeling

### [Tag My News](http://acube.di.unipi.it/tmn-dataset/)
TagMyNews Datasets is a collection of datasets of short text fragments which are used for topic-based text classifier. It is used in several other papers and is more difficult than the News Aggregator Dataset considering the scarcity of data and more categories.

## Dependency
### Download w2v from google
Download [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
Reference to [Google word2vec](https://code.google.com/archive/p/word2vec/)

### Install Keras
Reference to [Keras](https://keras.io/)
```
pip install keras
```

## Implementation
### Machine Learning Model on Different Sentence Representation
Training count of words on SVM and logistic regression
```
python bow_main.py (bow config file) (training data) (testing data)
```
Training w2v on SVM and logistic regression
```
python w2v_main.py (w2v config file) (google w2v pretrain model) (training data) (testing data)
```

### CNN
To train the CNN with bow of words use the following cmd
```
python cnn_main.py bow (bow config file) (training data) (testing data)
```
To train the CNN with word2vec use the following cmd
```
python cnn_main.py w2v (w2v config file) (training data) (valid data) (google w2v pretrain model)
```
