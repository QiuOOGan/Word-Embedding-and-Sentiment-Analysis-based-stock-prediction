import json

##### Create a json file with key of dates, the value of each dates is the array of articles in that day#####

f = open('news.json')
data = json.load(f)

date_to_articles = {}
for key in data:
    articles = data[key]
    for article in articles:
        date = article["pub_time"][:10]
        if date in date_to_articles:
            date_to_articles[date].append(article)
        else:
            date_to_articles[date] = [article]

with open('./json_files/date_to_articles_array.json', 'w') as fp:
    json.dump(date_to_articles, fp, sort_keys=True, indent=4)
