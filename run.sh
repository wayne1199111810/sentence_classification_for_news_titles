# Losgistic Regression and SVM
# BOW
python bow_main.py config/config_bow_a3.json ../data/trainCorpora_a3 ../data/validCorpora_a3
python bow_main.py config/config_bow_event.json ../data/trainCorpora_event ../data/validCorpora_event
python bow_main.py config/config_bow_uci.json ../data/trainCorpora_uci ../data/validCorpora_uci

# W2V
python w2v_main.py config/config_w2v_a3.json ../data/GoogleNews-vectors-negative300.bin ../data/trainCorpora_a3 ../data/validCorpora_a3
python w2v_main.py config/config_w2v_event.json ../data/GoogleNews-vectors-negative300.bin ../data/trainCorpora_event ../data/validCorpora_event
python w2v_main.py config/config_w2v_uci.json ../data/GoogleNews-vectors-negative300.bin ../data/trainCorpora_uci ../data/validCorpora_uci

# CNN
## bow
python cnn_main.py bow config/config_bow_a3.json ../data/trainCorpora_a3 ../data/validCorpora_a3
python cnn_main.py bow config/config_bow_event.json ../data/trainCorpora_event ../data/validCorpora_event
python cnn_main.py bow config/config_bow_uci.json ../data/trainCorpora_uci ../data/validCorpora_uci
## w2v
python cnn_main.py w2v config/config_w2v_a3.json ../data/trainCorpora_a3 ../data/validCorpora_a3 ../data/GoogleNews-vectors-negative300.bin
python cnn_main.py w2v config/config_w2v_event.json ../data/trainCorpora_event ../data/validCorpora_event ../data/GoogleNews-vectors-negative300.bin
python cnn_main.py w2v config/config_w2v_uci.json ../data/trainCorpora_uci ../data/validCorpora_uci ../data/GoogleNews-vectors-negative300.bin

## evaluateww
python evaluate_cnn.py bow config/config_bow_a3.json ../data/validCorpora_a3 model/w2v_a3.h5
python evaluate_cnn.py bow config/config_bow_event.json ../data/validCorpora_event model/w2v_event.h5
python evaluate_cnn.py bow config/config_bow_uci.json ../data/validCorpora_uci model/w2v_uci.h5

python evaluate_cnn.py w2v config/config_w2v_a3.json ../data/validCorpora_a3 ../data/GoogleNews-vectors-negative300.bin
python evaluate_cnn.py w2v config/config_w2v_event.json ../data/validCorpora_event ../data/GoogleNews-vectors-negative300.bin
python evaluate_cnn.py w2v config/config_w2v_uci.json ../data/validCorpora_uci ../data/GoogleNews-vectors-negative300.bin
