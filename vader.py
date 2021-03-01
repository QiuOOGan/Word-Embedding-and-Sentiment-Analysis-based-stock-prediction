from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sent = "VADER is smart, handsome, and funny."


# sents = ["Most automated sentiment analysis tools are shit.",
#          "VADER sentiment analysis is the shit.",
#          "Sentiment analysis has never been good.",
#          "VADER is smart, handsome, and funny."]

sid = SentimentIntensityAnalyzer()
# for sent in sents:
#     ss = sid.polarity_scores(sent)
#     print(ss)


# sent2 = "VADER is smart, handsome, and funny. Sentiment analysis has never been good.Most automated sentiment analysis tools are shit."
# ss = sid.polarity_scores(sent2)
# print(ss)

f = open('news.json')
data = json.load(f)
article_sample = data["FB"][0]
print("Article Sample", json.dumps(article_sample, indent=4, sort_keys=True))

counter = 0
date_to_company_to_arrayOfMethodsScores = {}
for company in data:
    articles_for_company = data[company]
    for article in articles_for_company:
        time = article["pub_time"][:10]
        score = sid.polarity_scores(article["text"])
        if time in date_to_company_to_arrayOfMethodsScores:
            if company in date_to_company_to_arrayOfMethodsScores[time]:
                date_to_company_to_arrayOfMethodsScores[time][company].append({"vader":score})
            else:
                date_to_company_to_arrayOfMethodsScores[time][company] = [{"vader":score}]
        else:
            date_to_company_to_arrayOfMethodsScores[time] = {}
            date_to_company_to_arrayOfMethodsScores[time][company] = [{"vader":score}]
        counter += 1
        print("done: ", counter)


with open('date_to_company_to_arrayOfMethodsScores.json', 'w') as fp:
    json.dump(date_to_company_to_arrayOfMethodsScores, fp, sort_keys=True, indent=4)
