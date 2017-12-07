SUMMARY: Dataset of references (urls) to news web pages

DESCRIPTION: Dataset of
# Dataset
	## UCI
		Reference to news web pages collected from an online aggregator in the period from March 10 to August 10 of 2014. The resources are grouped into clusters that represent pages discussing the same news story. The dataset includes also references to web pages that point (has a link to) one of the news page in the collection.
	## Event Registry
	## TagMyNews

# Download w2v from google
	Reference to [Google word2vec](https://code.google.com/archive/p/word2vec/)
	[GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

# Install Keras
	reference to [Keras](https://keras.io/)
	```
		pip install keras
	```
# BaseLine Model
	** Training count of words on SVM and logistic regression
		```
			python bow_main.py
		```
	** Training w2v on SVM and logistic regression
		```
			python w2v_main.py
		```

# CNN
	** Training on UCI dataset
		```
			python cnn_main.py bow model/cnn_bow.h5 LOG/cnn_bow_uci uci
		```
	** Training on Event
