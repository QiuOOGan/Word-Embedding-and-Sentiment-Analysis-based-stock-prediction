import json
from transformers import AutoModelForSequenceClassification
import nltk
import pandas as pd
from finbert.finbert import predict
import torch

print(torch.cuda.is_available())


def createFinbertJSON():
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

# createFinbertJSON()
# with open('date_to_company_to_arrayOfMethodsScores.json') as f:
#     vaderJSON = json.load(f)
# with open('finbert.json') as f:
#  finbertJSON = json.load(f)
#
#
# for date in vaderJSON.keys():
#     if date in finbertJSON:
#         for company in vaderJSON[date].keys():
#             arr_1 = vaderJSON[date][company]
#             arr_2 = finbertJSON[date][company]
#             for i in range((max(len(arr_1), len(arr_2)))):
#                 arr_1[i].update(arr_2[i])
#             vaderJSON[date][company] = arr_1
#
# with open('combined.json', 'w') as fp:
#     json.dump(vaderJSON, fp, sort_keys=True, indent=4)
# df = pd.read_csv('./features.csv')
# prices = pd.read_csv('./historical_price/AAPL_2015-12-30_2021-02-21_minute.csv')
# prices = prices[(prices.t >= df['date'][0])]
# temp = []
# time = []
# df['c'] = df.apply(lambda row : prices[(prices.t >= row.date)].iloc[0]['c'], axis=1)
# df = df.drop(columns=df.columns[0])
# df = df.drop(columns='date')
# # df['c'] = df.apply(lambda row : time.append(prices.apply(lambda p : (datetime.strptime(p['t'], '%Y-%m-%d %H:%M:%S'))
# #                                                                     >= datetime.strptime(row.date,'%Y-%m-%d %H:%M:%S'), axis=1).iloc[0]['t']), axis=1)
# #prices[(datetime.strptime(prices.t, '%y/%m/%d %H:%M:%S') >= datetime.strptime(row.date,'%y/%m/%d %H:%M:%S'))].iloc[0]['t']
# ts = 0.2
# ts = int(len(df)*ts)
# train = df[:-ts]
# test = df[-ts:]
#
# # Create Variables needed
# x_train = train.drop(columns='c')
# x_train = x_train.values
# y_train = train['c']
# x_test = test.drop(columns='c')
# x_test = x_test.values
# y_test = test['c']
#
# n = len(df.columns) - 1




