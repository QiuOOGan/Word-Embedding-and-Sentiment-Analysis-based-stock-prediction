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

def combine_prices():
    directory = os.path.join("./historical_price/")
    all_files = glob.glob(directory + "*")
    combined_prices = open('combined_prices.csv','a')
    isHeader = True
    headers = ""
    for file in all_files:
        temp = pd.read_csv(file)
        temp['company'] = file.split('\\')[-1].split('_')[0]
        temp = temp.drop(temp.columns[0], axis=1)
        headers = temp.columns
        temp['t'] = temp['t'].apply(lambda x : x[:10])
        dates = temp['t'].unique()
        filtered_data = []
        for date in dates:
            prices_of_date = temp[(temp.t == date)]
            prices_of_date = prices_of_date.sample(n=50 if len(prices_of_date) > 50 else len(prices_of_date), random_state=0)
            filtered_data.append(prices_of_date)
        temp = pd.concat(filtered_data).reset_index(drop=True)
        print(temp.head)
        temp.to_csv('temp.csv')
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
header = ['drop','v','vw','o','c','h','l','t','n','company']


def moodData():
    with open('date_to_moods.json') as f:
        moods = json.load(f)
    df = pd.read_csv('combined_prices.csv', names=header)
    df = df.drop('drop', axis=1)
    df['calm'] = df.apply(lambda x : moods.get(x['t'],[0,0,0,0])[0], axis=1)
    df['happy'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[1], axis=1)
    df['alert'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[2], axis=1)
    df['kind'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[3], axis=1)
    df = df.drop('t', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('mood.pkl')

def finberData():
    with open('finbert.json') as f:
        finbertJSON = json.load(f)
    df = pd.read_csv('combined_prices.csv', names=header)
    df = df.drop('drop', axis=1)
    def getFinData(x, dataName):
        try:
            lst = finbertJSON[x['t']][x['company']]
            result = 0
            for l in lst:
                result += l['finbert'][dataName]
            return result/len(lst)
        except KeyError:
            return 0
    df['finbert_negative'] = df.apply(lambda x : getFinData(x, 'negative'), axis=1)
    df['finbert_neutral'] = df.apply(lambda x: getFinData(x, 'neutral'), axis=1)
    df['finbert_positive'] = df.apply(lambda x: getFinData(x, 'positive'), axis=1)
    df['finbert_sentiment_score'] = df.apply(lambda x: getFinData(x, 'sentiment_score'), axis=1)
    df = df.drop('t', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('finbert.pkl')

def vaderData():
    with open('date_to_company_to_vader.json') as f:
        vaderJSON = json.load(f)
    df = pd.read_csv('combined_prices.csv', names=header)
    df = df.drop('drop', axis=1)
    def getVaderData(x, dataName):
        try:
            lst = vaderJSON[x['t']][x['company']]
            result = 0
            for l in lst:
                result += l['vader'][dataName]
            return result/len(lst)
        except KeyError:
            return 0
    df['vader_negative'] = df.apply(lambda x : getVaderData(x, 'neg'), axis=1)
    df['vader_neutral'] = df.apply(lambda x: getVaderData(x, 'neu'), axis=1)
    df['vader_positive'] = df.apply(lambda x: getVaderData(x, 'pos'), axis=1)
    df['vader_compound'] = df.apply(lambda x: getVaderData(x, 'compound'), axis=1)
    df = df.drop('t', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('vader.pkl')

def SRAFData():
    sentiments = {"sraf_negative": 0, "sraf_positive": 1, "sraf_uncertainty": 2, "sraf_litigious": 3,
                  "sraf_strongamodal": 4, "sraf_weakmodal": 5, "sraf_constraining": 6}

    with open('date_to_company_to_sraf.json') as f:
        srafJSON = json.load(f)
    df = pd.read_csv('combined_prices.csv', names=header)
    df = df.drop('drop', axis=1)
    def getSRAFData(x, dataName):
        try:
            lst = srafJSON[x['t']][x['company']]
            result = 0
            for l in lst:
                result += l['sraf'][sentiments[dataName]]
            return result/len(lst)
        except KeyError:
            return 0
        except IndexError:
            return 0

    for s in sentiments.keys():
        df[s] =  df.apply(lambda x : getSRAFData(x, s), axis=1)
    df = df.drop('t', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle('sraf.pkl')

def allData():
        with open('date_to_moods.json') as f:
            moods = json.load(f)
        with open('finbert.json') as f:
            finbertJSON = json.load(f)
        with open('date_to_company_to_vader.json') as f:
            vaderJSON = json.load(f)
        with open('date_to_company_to_sraf.json') as f:
            srafJSON = json.load(f)
        df = pd.read_csv('combined_prices.csv', names=header)
        df = df.drop('drop', axis=1)
        df['calm'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[0], axis=1)
        df['happy'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[1], axis=1)
        df['alert'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[2], axis=1)
        df['kind'] = df.apply(lambda x: moods.get(x['t'], [0, 0, 0, 0])[3], axis=1)
        def getFinData(x, dataName):
            try:
                lst = finbertJSON[x['t']][x['company']]
                result = 0
                for l in lst:
                    result += l['finbert'][dataName]
                return result / len(lst)
            except KeyError:
                return 0
        df['finbert_negative'] = df.apply(lambda x: getFinData(x, 'negative'), axis=1)
        df['finbert_neutral'] = df.apply(lambda x: getFinData(x, 'neutral'), axis=1)
        df['finbert_positive'] = df.apply(lambda x: getFinData(x, 'positive'), axis=1)
        df['finbert_sentiment_score'] = df.apply(lambda x: getFinData(x, 'sentiment_score'), axis=1)
        def getVaderData(x, dataName):
            try:
                lst = vaderJSON[x['t']][x['company']]
                result = 0
                for l in lst:
                    result += l['vader'][dataName]
                return result / len(lst)
            except KeyError:
                return 0
        df['vader_negative'] = df.apply(lambda x: getVaderData(x, 'neg'), axis=1)
        df['vader_neutral'] = df.apply(lambda x: getVaderData(x, 'neu'), axis=1)
        df['vader_positive'] = df.apply(lambda x: getVaderData(x, 'pos'), axis=1)
        sentiments = {"sraf_negative": 0, "sraf_positive": 1, "sraf_uncertainty": 2, "sraf_litigious": 3,
                      "sraf_strongamodal": 4, "sraf_weakmodal": 5, "sraf_constraining": 6}

        def getSRAFData(x, dataName):
            try:
                lst = srafJSON[x['t']][x['company']]
                result = 0
                for l in lst:
                    result += l['sraf'][sentiments[dataName]]
                return result / len(lst)
            except KeyError:
                return 0
            except IndexError:
                return 0
        for s in sentiments.keys():
            df[s] = df.apply(lambda x: getSRAFData(x, s), axis=1)
        df = df.drop('t', axis=1)
        df = df.drop('company', axis=1)
        df = df.sample(frac=1, random_state=0)
        df.to_pickle('alldata.pkl')

# moodData()
# finberData()
# vaderData()
# SRAFData()

df = pd.read_pickle('sraf.pkl')
print(df.head)
df = df.dropna()
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




