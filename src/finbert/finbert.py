import json
from transformers import AutoModelForSequenceClassification
import nltk
import pandas as pd
from finbert.finbert import predict
import torch

nltk.download('punkt')
model = AutoModelForSequenceClassification.from_pretrained('./models/classifier_model/finbert-sentiment',
                                                           num_labels=3, cache_dir=None)

with open('news.json') as f:
    data = json.load(f)

finbertJSON = {}
counter = 0
keys = list(data.keys())
for company in keys:
    print((counter/len(data.keys())) * 100)
    counter+=1
    counter_2 = 0
    articles = data[company]
    for article in articles:
        print(company + ': ', (counter_2 / len(articles)) * 100)
        counter_2 += 1
        # if article['pub_time'][:10] != '2016-01-28':
        #     continue
        temp = {}
        if article['pub_time'][:10] in finbertJSON.keys():
            temp = finbertJSON[article['pub_time'][:10]]
        article_arr = []
        if company in temp.keys():
            article_arr = temp[company]
        a = {'positive' : 0, 'negative' : 0, 'neutral' : 0, 'sentiment_score' : 0}
        text = article['text']
        text = text.replace('\t', '')
        text = text.replace('\0', '')
        finbert_score = predict(text=text, model=model, write_to_csv=False, path=None)
        if len(finbert_score) != 0:
            logit_avg = finbert_score['logit'].sum()/len(finbert_score)
            a['positive'] = logit_avg[0].item()
            a['negative'] = logit_avg[1].item()
            a['neutral'] = logit_avg[2].item()
            a['sentiment_score'] = (finbert_score['sentiment_score'].sum()/len(finbert_score)).item()
        f = {}
        f['finbert'] = a
        article_arr.append(f)
        temp[company] = article_arr
        #add more features here
        finbertJSON[article['pub_time'][:10]] = temp

with open('finbert.json', 'w') as fp:
    json.dump(finbertJSON, fp, sort_keys=True, indent=4)