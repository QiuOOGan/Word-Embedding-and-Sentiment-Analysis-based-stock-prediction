import json
from transformers import AutoModelForSequenceClassification
import nltk
from finbert.finbert import predict
import torch
import os
import glob
import pandas as pd
import numpy as np

np.random.seed(0)
header = ["day " + str(int(i / 2) + 1) if i % 2 == 1 else "day " + str(int(i / 2)) + " date" for i in
                range(1, 63)]

def combine_prices():
    directory = os.path.join("./historical_price/")
    all_files = glob.glob(directory + "*")
    isHeader = True
    for file in all_files:
        combined_prices = open('../combined_prices_rl.csv', 'a')
        temp = pd.read_csv(file)
        filename = "./historical_price_df/" + file.split('\\')[-1].split('_')[0] + ".pkl"
        temp['t'] = temp['t'].apply(lambda x: x[:10])
        dates = temp['t'].unique()
        filtered_data = []
        for date in dates:
            prices_of_date = temp[(temp.t == date)]
            # prices_of_date = prices_of_date.sample(n=50 if len(prices_of_date) > 50 else len(prices_of_date), random_state=0)
            filtered_data.append(prices_of_date.iloc[[0]])
        temp = pd.concat(filtered_data).reset_index(drop=True)
        closes = temp['c'].values
        data = []
        for i in range(30, len(closes)):
            arr = []
            for j in range(i-30, i+1):
                arr.append(closes[j])
                arr.append(dates[j])
            data.append(np.array(arr))
        data = pd.DataFrame(data, columns=header)
        data['company'] = file.split('\\')[-1].split('_')[0]
        data.to_csv('temp.csv')
        with open('temp.csv', 'r') as f:
            for line in f:
                if isHeader:
                    isHeader = False
                    continue
                combined_prices.write(line)
            isHeader = True
        combined_prices.close()
    # df = df.drop(df.columns[0], axis=1)
    # df = df.dropna()
    # print(df.head)
# combine_prices()
header.insert(0,'drop')
header.append('company')


def moodData():
    with open('../date_to_moods.json') as f:
        moods = json.load(f)
    df = pd.read_csv('../combined_prices_rl.csv', names=header)
    df = df.drop('drop', axis=1)
    columns = df.columns

    for i in range(1, 31):
        date = columns[2 * i - 1]
        df[str(i) + 'calm'] = df.apply(lambda x : moods.get(x[date],[0,0,0,0])[0], axis=1)
        df[str(i) + 'happy'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[1], axis=1)
        df[str(i) + 'alert'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[2], axis=1)
        df[str(i) + 'kind'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[3], axis=1)
        df = df.drop(date, axis=1)
    df = df.drop('day 31 date', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('mood.pkl')

def finberData(summarize = False):
    filename = 'finbert_with_summarize' if summarize else 'finbert'
    with open(filename + '.json') as f:
        finbertJSON = json.load(f)
    df = pd.read_csv('../combined_prices_rl.csv', names=header)
    df = df.drop('drop', axis=1)
    def getFinData(x, date, dataName):
        try:
            lst = finbertJSON[x[date]][x['company']]
            result = 0
            for l in lst:
                result += l['finbert'][dataName]
            return result/len(lst)
        except KeyError:
            return 0

    columns = df.columns
    for i in range(1, 31):
        date = columns[2 * i - 1]
        # df['finbert_negative'] = df.apply(lambda x : getFinData(x, date, 'negative'), axis=1)
        # df['finbert_neutral'] = df.apply(lambda x: getFinData(x, date,'neutral'), axis=1)
        # df['finbert_positive'] = df.apply(lambda x: getFinData(x,date, 'positive'), axis=1)
        df['finbert_sentiment_score'] = df.apply(lambda x: getFinData(x,date, 'sentiment_score'), axis=1)
    df = df.drop('day 31 date', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle(filename + '.pkl')

def vaderData():
    with open('../date_to_company_to_vader.json') as f:
        vaderJSON = json.load(f)
    df = pd.read_csv('../combined_prices_rl.csv', names=header)
    df = df.drop('drop', axis=1)
    def getVaderData(x,date, dataName):
        try:
            lst = vaderJSON[x[date]][x['company']]
            result = 0
            for l in lst:
                result += l['vader'][dataName]
            return result/len(lst)
        except KeyError:
            return 0
    columns = df.columns
    for i in range(1, 31):
        date = columns[2 * i - 1]
        # df['vader_negative'] = df.apply(lambda x : getVaderData(x,date, 'neg'), axis=1)
        # df['vader_neutral'] = df.apply(lambda x: getVaderData(x,date, 'neu'), axis=1)
        # df['vader_positive'] = df.apply(lambda x: getVaderData(x,date, 'pos'), axis=1)
        df['vader_compound'] = df.apply(lambda x: getVaderData(x,date, 'compound'), axis=1)
    df = df.drop('day 31 date', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('vader.pkl')

def SRAFData():
    sentiments = {"sraf_negative": 0, "sraf_positive": 1, "sraf_uncertainty": 2, "sraf_litigious": 3,
                  "sraf_strongamodal": 4, "sraf_weakmodal": 5, "sraf_constraining": 6}

    with open('../date_to_company_to_sraf.json') as f:
        srafJSON = json.load(f)
    df = pd.read_csv('../combined_prices_rl.csv', names=header)
    df = df.drop('drop', axis=1)
    def getSRAFData(x,date, dataName):
        try:
            lst = srafJSON[x[date]][x['company']]
            result = 0
            for l in lst:
                result += l['sraf'][sentiments[dataName]]
            return result/len(lst)
        except KeyError:
            return 0
        except IndexError:
            return 0
    columns = df.columns
    for i in range(1, 31):
        date = columns[2 * i - 1]
        for s in sentiments.keys():
            df[s] =  df.apply(lambda x : getSRAFData(x,date, s), axis=1)
    df = df.drop('day 31 date', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('sraf.pkl')

def allData():
        with open('../date_to_moods.json') as f:
            moods = json.load(f)
        with open('../finbert_with_summarize.json') as f:
            finbertJSON = json.load(f)
        with open('../date_to_company_to_vader.json') as f:
            vaderJSON = json.load(f)
        with open('../date_to_company_to_sraf.json') as f:
            srafJSON = json.load(f)
        df = pd.read_csv('../combined_prices_rl.csv', names=header)
        df = df.drop('drop', axis=1)
        sentiments = {"sraf_negative": 0, "sraf_positive": 1, "sraf_uncertainty": 2, "sraf_litigious": 3,
                      "sraf_strongamodal": 4, "sraf_weakmodal": 5, "sraf_constraining": 6}
        def getSRAFData(x, date, dataName):
            try:
                lst = srafJSON[x[date]][x['company']]
                result = 0
                for l in lst:
                    result += l['sraf'][sentiments[dataName]]
                return result / len(lst)
            except KeyError:
                return 0
            except IndexError:
                return 0

        def getFinData(x, date, dataName):
            try:
                lst = finbertJSON[x[date]][x['company']]
                result = 0
                for l in lst:
                    result += l['finbert'][dataName]
                return result / len(lst)
            except KeyError:
                return 0

        def getVaderData(x, date, dataName):
            try:
                lst = vaderJSON[x[date]][x['company']]
                result = 0
                for l in lst:
                    result += l['vader'][dataName]
                return result / len(lst)
            except KeyError:
                return 0

        columns = df.columns
        for i in range(1, 31):
            date = columns[2 * i - 1]
            df[str(i) + 'calm'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[0], axis=1)
            df[str(i) + 'happy'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[1], axis=1)
            df[str(i) + 'alert'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[2], axis=1)
            df[str(i) + 'kind'] = df.apply(lambda x: moods.get(x[date], [0, 0, 0, 0])[3], axis=1)
            df['finbert_sentiment_score'] = df.apply(lambda x: getFinData(x,date, 'sentiment_score'), axis=1)
            df['vader_compound'] = df.apply(lambda x: getVaderData(x, date, 'compound'), axis=1)
            for s in sentiments.keys():
                df[s] = df.apply(lambda x: getSRAFData(x,date, s), axis=1)

        df = df.drop('day 31 date', axis=1)
        df = df.drop('company', axis=1)
        df = df.sample(frac=1, random_state=0)
        df.to_pickle('alldata.pkl')

# moodData()
# finberData()
# vaderData()
# SRAFData()
# finberData(summarize=True)
# allData()
methods = ['mood','finbert','finbert_with_summarize','vader','sraf','alldata']

# finbert = pd.read_pickle('finbert_with_summarize.pkl')
# vader = pd.read_pickle('vader.pkl')
# df = pd.read_pickle('sraf.pkl')
# mood = pd.read_pickle('mood.pkl')
# df['finbert_negative'] = pd.Series(finbert['finbert_negative'])
# df['finbert_positive'] = pd.Series(finbert['finbert_positive'])
# df['finbert_neutral'] = pd.Series(finbert['finbert_neutral'])
# df['vader_negative'] = pd.Series(vader['vader_negative'])
# df['vader_positive'] = pd.Series(vader['vader_positive'])
# df['vader_neutral'] = pd.Series(vader['vader_neutral'])
# df['vader_compound'] = pd.Series(vader['vader_compound'])
# df['calm'] = pd.Series(mood['calm'])
# df['kind'] = pd.Series(mood['kind'])
# df['alert'] = pd.Series(mood['alert'])
# df['happy'] = pd.Series(mood['happy'])
# df.to_pickle('alldata.pkl')



# ts = 0.2
# ts = int(len(df)*ts)
# train = df[:-ts]
# test = df[-ts:]
#
# # Create Variables needed
# x_train = train.drop(columns='day 31')
# x_train = x_train.values
# y_train = train['day 31']
# x_test = test.drop(columns='day 31')
# x_test = x_test.values
# y_test = test['day 31']

def saveToLSTMData(x):
    datapoint = []
    for i in range(1, 31):
        timestamp = []
        timestamp.append(x['day ' + str(i)])
        for j in range(0, sentiment_feature_count):
            timestamp.append(x[df.columns[i + 30 + j]])
        datapoint.append(timestamp)
    data_x.append(datapoint)
    data_y.append(x['day 31'])

for method_name in methods:
    df = pd.read_pickle(method_name + '.pkl')
    print(df.head)
    df = df.dropna()
    sentiment_feature_count = int((len(df.columns) - 31)/30)
    data_x = []
    data_y = []

    df.apply(lambda x : saveToLSTMData(x), axis=1)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    with open('./LSTM_data/' + method_name + '_x' + '.npy', 'wb') as f:
        np.save(f, data_x)
    with open('./LSTM_data/' + method_name + '_y' + '.npy', 'wb') as f:
        np.save(f, data_y)



