# bow
python bow_main.py
python cnn_main.py bow model/cnn_bow.h5 LOG/cnn_bow_api api
python cnn_main.py bow model/cnn_bow.h5 LOG/cnn_bow_uci uci

# w2v
python w2v_main.py
python cnn_main.py w2v model/cnn_w2v.h5 LOG/cnn_w2v_api api
python cnn_main.py w2v model/cnn_w2v.h5 LOG/cnn_w2v_uci uci

# W2V
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci

# python w2v_main.py uci ../data/sentence_classification_for_news_titles/trainCorpora.csv ../data/sentence_classification_for_news_titles/validCorpora.csv

# BOW
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci
# python bow_main.py uci ../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv ../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv

# cnn
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_bow_a3.json LOG/cnn_bow_a3 model/bow_a3
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_bow_event.json LOG/cnn_bow_event model/bow_event
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci config/config_bow_uci.json LOG/cnn_bow_uci model/bow_uci


python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_w2v_a3.json LOG/cnn_w2v_a3 model/w2v_a3
python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_w2v_event.json LOG/cnn_w2v_event model/w2v_event
python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci config/config_w2v_uci.json LOG/cnn_w2v_uci model/w2v_uci
