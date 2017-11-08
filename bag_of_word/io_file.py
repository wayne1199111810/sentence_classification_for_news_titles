import pandas as pd


def read_corpora_file(file_name):
    df = pd.read_csv(file_name, sep='	', names=['ID', 'title', 'url', 'publisher',
                                                   'category', 'story', 'hostname', 'timestamp'])
    return df


def split_train_dev_test(df):
    dev_start = 296056
    test_start = 359497
    return df.iloc[:dev_start, :], \
           df.iloc[dev_start:test_start].reset_index(drop=True), \
           df.iloc[test_start:].reset_index(drop=True)
