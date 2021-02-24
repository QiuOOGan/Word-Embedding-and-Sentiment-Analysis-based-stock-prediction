import json
import pandas as pd
###################### news.json ######################
f = open('news.json')
data = json.load(f)
print("Company Count: ", len(data.keys()))
print("Company Name: ", data.keys())
article_sample = data["FB"][0]
print("Article Sample", json.dumps(article_sample, indent=4, sort_keys=True))

FB = data["FB"]
for i in range(0,10):
    article_sample = FB[i]
    time = FB[i]["pub_time"]
    text = FB[i]["text"]
    print(time)
    print(text[:50])


###################### historical_price/FB ######################
df = pd.read_csv("historical_price/FB_2015-12-30_2021-02-21_minute.csv")
columns = list(df.columns)

print(df.head(5))
print(df["v"][0])
print(columns)