import os, csv

data_dir = './processed_data'

train_file = '../data/sentence_classification_for_news_titles/api_data/corpora/trainCorpora.csv'
valid_file = '../data/sentence_classification_for_news_titles/api_data/corpora/validCorpora.csv'

if __name__ == '__main__':
	file_names = os.listdir(data_dir)
	text = set()
	count = 0
	for file_name in file_names:
		path = data_dir + '/' + file_name
		with open(path, 'r', encoding="utf-8") as f:
			for line in f:
				count += 1
				text.add(line)
	writefile = 'news_from_EventRegist.csv'
	with open(writefile, 'w', encoding="utf-8", newline='') as f:
		csv_writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_ALL)
		for line in text:
			line = line.strip()
			idx = line.index(',')
			data = list(["{} {}".format(line[0:idx],line[idx+1:])])
			csv_writer.writerow(data)