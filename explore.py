import json
import pandas as pd
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
import nltk
from finbert.finbert import predict
nltk.download('punkt')
###################### news.json ######################
f = open('news.json')
data = json.load(f)
print("Company Count: ", len(data.keys()))
print("Company Name: ", data.keys())
article_sample = data["FB"][0]
print("Article Sample", json.dumps(article_sample, indent=4, sort_keys=True))

model = AutoModelForSequenceClassification.from_pretrained('./models/classifier_model/finbert-sentiment', num_labels=3, cache_dir=None)
FB = data["FB"]
for i in range(0,10):
    print("article %i", i)
    article_sample = FB[i]
    time = FB[i]["pub_time"]
    text = FB[i]["text"]
    print(time)
    print(text[:50])
    text = text.replace('\t','')
    text = text.replace('\0', '')
    trial = "Shares in the spin-off of South African e-commerce group Naspers surged more than 25% in the first minutes of their market debut in Amsterdam on Wednesday."
    print(predict(text=text, model=model, write_to_csv=False, path=None))
    # print(predict(text, model).to_json(orient='records'))


###################### historical_price/FB ######################
df = pd.read_csv("historical_price/FB_2015-12-30_2021-02-21_minute.csv")
columns = list(df.columns)

print(df.head(5))
print(df["v"][0])
print(columns)