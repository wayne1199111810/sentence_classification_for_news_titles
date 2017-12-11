import pandas as pd
import os
import csv

def read_corpora_file(file_name):
    df = pd.read_csv(file_name, sep='	', names=['ID', 'title', 'url', 'publisher',
                                                   'category', 'story', 'hostname', 'timestamp'],
                     chunksize=500,
                     dtype=object)
    return df

def process_aggregator_file(file_name):
    print('processing {0}'.format(file_name))
    chunks = read_corpora_file(file_name)
    out_file_name = os.path.splitext(file_name)[0]
    with open(out_file_name, 'w') as f:
        for chunk in chunks:
            out_data = chunk.loc[:, ['category', 'title']]
            for _, row in out_data.iterrows():
                f.write(' '.join([row['category'], row['title']]) + '\n')
            #out_data.to_csv(f, sep=' ', index=False, header=False,
            #                quoting=csv.QUOTE_NONE, quotechar='', escapechar=' ')

if __name__ == "__main__":
    file_list = ['data/trainCorpora.csv', 'data/validCorpora.csv', 'data/testCorpora.csv']
    for file_str in file_list:
        process_aggregator_file(file_str)
