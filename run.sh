# Losgistic Regression and SVM
# BOW
python bow_main.py config/config_bow_a3.json data/a3/trainCorpora data/a3/validCorpora
python bow_main.py config/config_bow_event.json data/event/trainCorpora data/event/validCorpora
python bow_main.py config/config_bow_uci.json data/uci/trainCorpora data/uci/validCorpora

# W2V
python w2v_main.py config/config_w2v_a3.json ../data/GoogleNews-vectors-negative300.bin data/a3/trainCorpora data/a3/validCorpora
python w2v_main.py config/config_w2v_event.json ../data/GoogleNews-vectors-negative300.bin data/event/trainCorpora data/event/validCorpora
python w2v_main.py config/config_w2v_uci.json ../data/GoogleNews-vectors-negative300.bin data/uci/trainCorpora data/uci/validCorpora

# CNN
## bow
python cnn_main.py bow config/config_bow_a3.json data/a3/trainCorpora data/a3/validCorpora
python cnn_main.py bow config/config_bow_event.json data/event/trainCorpora data/event/validCorpora
python cnn_main.py bow config/config_bow_uci.json data/uci/trainCorpora data/uci/validCorpora
## w2v
python cnn_main.py w2v config/config_w2v_a3.json data/a3/trainCorpora data/a3/validCorpora ../data/GoogleNews-vectors-negative300.bin
python cnn_main.py w2v config/config_w2v_event.json data/event/trainCorpora data/event/validCorpora ../data/GoogleNews-vectors-negative300.bin
python cnn_main.py w2v config/config_w2v_uci.json data/uci/trainCorpora data/uci/validCorpora ../data/GoogleNews-vectors-negative300.bin

## evaluateww
python evaluate_cnn.py bow config/config_bow_a3.json data/a3/validCorpora model/w2v_a3.h5
python evaluate_cnn.py bow config/config_bow_event.json data/event/validCorpora model/w2v_event.h5
python evaluate_cnn.py bow config/config_bow_uci.json data/uci/validCorpora model/w2v_uci.h5

python evaluate_cnn.py w2v config/config_w2v_a3.json data/a3/validCorpora ../data/GoogleNews-vectors-negative300.bin
python evaluate_cnn.py w2v config/config_w2v_event.json data/event/validCorpora ../data/GoogleNews-vectors-negative300.bin
python evaluate_cnn.py w2v config/config_w2v_uci.json data/uci/validCorpora ../data/GoogleNews-vectors-negative300.bin
