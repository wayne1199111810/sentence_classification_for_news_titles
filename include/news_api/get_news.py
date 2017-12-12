from eventregistry import *
import json
# crawling the data from event registry API

import datetime
import os

def read_config(file_name):
    with open(file_name, 'r') as f:
        params = json.load(f)
    return params


if __name__ == "__main__":
    data_dir = "download_data/"
    config = read_config('./config/eventRegistry_key.json')

    er = EventRegistry(apiKey=config["apiKey"])

    categories = ['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Recreation', 'Reference', 'Regional',
                  'Science', 'Shopping', 'Society', 'Sports']
    cur_day = datetime.date.today()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    downloaded_file = os.listdir(data_dir)
    dates = 1
    max_items = 5000

    for prev_days, category in enumerate(categories):
        start_date = cur_day + datetime.timedelta(days=-prev_days)
        end_date = cur_day + datetime.timedelta(days=1-prev_days)
        file_name = category + '_' + str(start_date) + '_' + str(end_date) + ".news.out"

        print("category_start_date_end_date: {0}".format(file_name))

        if file_name in downloaded_file:
            print("already downloaded, skip")
            continue
        file_name = os.path.join(data_dir, file_name)
        q = QueryArticlesIter(
            dateStart=start_date, dateEnd=end_date,
            categoryUri=er.getCategoryUri(category),
            lang="eng")

        return_info = ReturnInfo(
                articleInfo=ArticleInfoFlags(body=False, categories=True, eventUri=False))

        with open(file_name, 'w', encoding="utf-8") as f:
            res = [article for article in q.execQuery(er, sortBy="date", maxItems=max_items,
                                                      returnInfo=return_info)]
            f.write(json.dumps(res))
        print("get {0} news".format(len(res)))
