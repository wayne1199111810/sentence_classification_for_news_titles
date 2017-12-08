# W2V
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event
python w2v_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci

# BOW
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event
python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci

python bow_main.py api ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_origin_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_origin_uci

# python bow_main.py uci ../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv ../data/sentence_classification_for_news_titles/newsCorpora.shuffled.csv

# cnn
## bow
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_bow_a3.json LOG/cnn_bow_a3 model/bow_a3.h5
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_bow_event.json LOG/cnn_bow_event model/bow_event.h5
python cnn_main.py api bow ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci config/config_bow_uci.json LOG/cnn_bow_uci model/bow_uci.h5

## w2v
python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_a3 ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_w2v_a3.json LOG/cnn_w2v_a3 model/w2v_a3.h5
python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_event ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_w2v_event.json LOG/cnn_w2v_event model/w2v_event.h5
python cnn_main.py api w2v ../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora_uci ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci config/config_w2v_uci.json LOG/cnn_w2v_uci model/w2v_uci.h5

## evaluate
python evaluate_cnn.py ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_w2v_a3.json model/w2v_a3.h5
python evaluate_cnn.py ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_w2v_event.json model/w2v_event.h5
python evaluate_cnn.py ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_uci config/config_w2v_uci.json model/w2v_uci.h5

python evaluate_cnn.py ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_a3 config/config_bow_a3.json model/bow_a3.h5
python evaluate_cnn.py ../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora_event config/config_bow_event.json model/bow_event.h5
