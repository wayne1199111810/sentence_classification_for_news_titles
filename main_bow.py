from include.getApiData import getApiData as api
from include import bag_of_word as BoW
import sys

if __name__ == '__main__':
	data_type = sys.argv[1]
	if data_type == 'api':
		runApi()
	elif data_type == 'UCI':
		run