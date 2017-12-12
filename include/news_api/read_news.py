"""
convert json to csv file with highest score as the main category
"""

import json
import csv
import os

data_dir = "download_data/"
output_dir = "processed_data/"

def get_highest_category(record):
    categories = record['categories']
    highest_category = max(categories, key=lambda x: x['wgt'])
    return parse_main_category(highest_category['uri']), record['title'].strip('\n')

def parse_main_category(category_uri):
    return category_uri.split('/')[1]

if __name__ == "__main__":
    file_names = os.listdir(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in file_names:
        print("reading file: {0}".format(file_name))
        with open(data_dir + file_name, encoding="utf-8") as f:
            news = json.loads(f.read())
            res = map(get_highest_category, news)
            with open(os.path.join(output_dir, file_name + '.csv'), 'w', encoding="utf-8", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')

                for row in res:
                    csv_writer.writerow(row)

        print("get {0} news".format(len(news)))
