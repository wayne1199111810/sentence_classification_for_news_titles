# bow
python bow_main.py
python cnn_main.py bow model/cnn_bow.h5 LOG/cnn_bow_api api
python cnn_main.py bow model/cnn_bow.h5 LOG/cnn_bow_uci uci
# w2v
python w2v_main.py
python cnn_main.py w2v model/cnn_w2v.h5 LOG/cnn_w2v_api api
python cnn_main.py w2v model/cnn_w2v.h5 LOG/cnn_w2v_uci uci
