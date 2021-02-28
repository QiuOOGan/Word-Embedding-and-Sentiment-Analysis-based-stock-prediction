import json
from transformers import AutoModelForSequenceClassification
import nltk
import pandas as pd
from finbert.finbert import predict
import torch

print(torch.cuda.is_available())


def createFeatureFrame():
    nltk.download('punkt')
    model = AutoModelForSequenceClassification.from_pretrained('./models/classifier_model/finbert-sentiment',
                                                               num_labels=3, cache_dir=None)

    with open('date_to_articles_array.json') as f:
        data = json.load(f)

    df = []
    counter = 0
    keys = list(data.keys())
    for date in keys[:5]:
        print((counter/len(data.keys())) * 100)
        counter+=1
        temp = []
        articles = data[date]
        for article in articles:
            text = article['text']
            text = text.replace('\t', '')
            text = text.replace('\0', '')
            finbert_score = sum(predict(text=text, model=model, write_to_csv=False, path=None)['sentiment_score'])
            a = {}
            a['date'] = article['pub_time'][:-6]
            a['finbert_score'] = finbert_score
            #add more features here
            df.append(a)

    df = pd.DataFrame.from_dict(df, orient='columns')
    df.to_csv('features.csv')
    print(df.head())

# createFeatureFrame()
df = pd.read_csv('./features.csv')
prices = pd.read_csv('./historical_price/AAPL_2015-12-30_2021-02-21_minute.csv')
prices = prices[(prices.t >= df['date'][0])]
temp = []
time = []
df['c'] = df.apply(lambda row : prices[(prices.t >= row.date)].iloc[0]['c'], axis=1)
df = df.drop(columns=df.columns[0])
df = df.drop(columns='date')
# df['c'] = df.apply(lambda row : time.append(prices.apply(lambda p : (datetime.strptime(p['t'], '%Y-%m-%d %H:%M:%S'))
#                                                                     >= datetime.strptime(row.date,'%Y-%m-%d %H:%M:%S'), axis=1).iloc[0]['t']), axis=1)
#prices[(datetime.strptime(prices.t, '%y/%m/%d %H:%M:%S') >= datetime.strptime(row.date,'%y/%m/%d %H:%M:%S'))].iloc[0]['t']
ts = 0.2
ts = int(len(df)*ts)
train = df[:-ts]
test = df[-ts:]

# Create Variables needed
x_train = train.drop(columns='c')
x_train = x_train.values
y_train = train['c']
x_test = test.drop(columns='c')
x_test = x_test.values
y_test = test['c']

n = len(df.columns) - 1




