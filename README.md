# Word Embedding and Sentiment Analysis Based Stock Prediction
# [Our Presentation Video](https://youtu.be/POyRvvsHKbI)
## Introduction:
In this project, we used several word embedding methods on the financial news dataset to create additional features to the prehistorical stock price data set and trained a model to predict the close price at a specific time. We formulate the problem as a regression problem. We first construct time series data and use the previous 30 day's data to predict the next day's closing price. The main tools we used are based on a language model called "BERT", i.e Bidirectional Encoder Representations from Transformers. In this project, we will train the model using **LSTM** with each of the tools listed below and compare their results. In the end, we will try to ensemble the tools to train a final model, and hopefully to get a better result.

## BERT
* BERT is a Transformer-based machine learning technique for NLP. There are two phases BERT that uses to solve problem:
  * Pretraining
  * Fine-Tunning for a specific task
* Take an example of finBERT: finBERT is a further trained BERT by fine-tunning it in the financial domain. 
Therefore, finBERT can be very helpful on our financial news dataset. The usage of BERT on our project will be explained in more details below.

## Tools:
* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
    * [finBERT](https://github.com/ProsusAI/finBERT)
    * [Sentence Transfermers](https://github.com/UKPLab/sentence-transformers)
* [fasttext](https://fasttext.cc/)
* [NLTK Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
* [Goel and Mittal's Paper](http://cs229.stanford.edu/proj2011/GoelMittal-StockMarketPredictionUsingTwitterSentimentAnalysis.pdf)
* [Software Repository for Accounting and Finance](https://sraf.nd.edu/textual-analysis/)

## Prerequisites
* 1 Install [finBERT](https://github.com/ProsusAI/finBERT) as the repository suggested. We set up our environment using their
 environment.yml. Also, you need the model downloaded from this link: [model's link](https://huggingface.co/ProsusAI/finbert)
* 2 Install [Sentence Transformers](https://github.com/UKPLab/sentence-transformers).
* 3 Vader. Install nltk
``` sh
 pip install nltk
```
* 4 [fasttext](https://fasttext.cc). Specifically, you need the model: wiki-news-300d-1M.vec.zip from this [link](https://fasttext.cc/docs/en/english-vectors.html)
* 5 gensim
``` sh
pip install gensim
```
* 6 [SRAF: Loughran-McDonald Sentiment Word Lists](https://sraf.nd.edu/textual-analysis/resources/)

## Steps to train our model
* 1. Make sure you have historical_price folder, date_to_company_to_sraf.json, date_to_company_to_vader.json,
date_to_moods.json, finbert.json, finbert_with_summarize.json in json_files directory. These files will be used to produce LSTM data.
  2. Run ./src/preprocessing.py. This step can take hours as it needs to generate the rolling prices first. Feel free to contact us for a copy of
  those data to skip this part.
  3. run ./src/lstm/train_lstm.py. Specify which method you want to use. 
* If you want to further improve upon our json data, make sure to have the related files such as fasttext word vector.

## Dataset:
* news.json:
it contains news articles from 81 big companies. Each company has an array of articles with field of "title", "text", "pul_time", and "url". Here is an example of "FaceBook":
  ```sh
  {
    "FB": 
        [
            {
                "titles": STRING,
                "url": STRING,
                "text": STRING,
                "pul_time": STRING
            }
        ]
        ...
  }
  ```
* historical_price/[company_name]_2015-12-30_2021-02-21_minute.csv
    They contain minute-level historical prices in the past 5 years for 86 companies. Here is what they look like:
    ```sh
    df = pd.read_csv("historical_price/FB_2015-12-30_2021-02-21_minute.csv")
    print(df.head(5))
    
       Unnamed: 0      v        vw  ...       l                    t    n
    0      945359  200.0  107.1000  ...  107.10  2015-12-30 09:24:00  2.0
    1      945358  130.0  106.8500  ...  106.85  2015-12-30 10:20:00  2.0
    2      945357  286.0  106.8308  ...  106.80  2015-12-30 10:30:00  4.0
    3      945356  100.0  106.8700  ...  106.87  2015-12-30 11:52:00  1.0
    4      945355  200.0  106.9900  ...  106.99  2015-12-30 11:58:00  1.0
    ```
## Data Preparation (Run ./src/preprocessing.py. Takes hours.):
* We formed time series data using only closing price column and time column. Because the size of prehistorical price is too big while the articles are too sparse, we decide to extract only one closing price of a day. The other option might be summarizing the closing price of that day by taking the average. We then form time series data and formulate the problem as a regression: use the previous 30 days of data to predict the next days closing price. 
* We used all the data in the new.json.
```sh
def combine_prices():
    directory = os.path.join("./historical_price/")
    all_files = glob.glob(directory + "*")
    isHeader = True
    for file in all_files:
        combined_prices = open('./combined_prices_rl.csv', 'a')
        temp = pd.read_csv(file)
        temp['t'] = temp['t'].apply(lambda x: x[:10])
        dates = temp['t'].unique()
        filtered_data = []
        for date in dates:
            prices_of_date = temp[(temp.t == date)]
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
   ```
* The result looks like the following:
```
       day 1    day 1 date    day 2   ...     day 31   day 31 date  company
0      43.2100  2015-12-30  42.8200   ...    35.4400   2016-02-09     AAL
1      42.8200  2015-12-31  41.0900   ...    36.2500   2016-02-10     AAL
2      41.0900  2016-01-04  41.0000   ...    36.9300   2016-02-11     AAL
3      41.0000  2016-01-05  40.6200   ...    36.7700   2016-02-12     AAL
4      40.6200  2016-01-06  41.4800   ...    37.8100   2016-02-13     AAL
5      41.4800  2016-01-07  40.3500   ...    37.8500   2016-02-16     AAL
6      40.3500  2016-01-08  40.3800   ...    38.3300   2016-02-17     AAL
```

## [finBERT](https://github.com/ProsusAI/finBERT)
#### Setup
* We start off by cloning the repo into a local directory and create a the corresponding conda environment with the necessary
packages. The conda command is:
   ```sh
        conda env create -f environment.yml
   ```
  In our case, we don't need to create a flask server and only need the main functionality of predicting. 
* We use the pretrained model provided by the team([link](https://huggingface.co/ProsusAI/finbert)). Make sure to have the
correct directory setup. Our current setup put the model under /models/classifier_model/finbert-sentiment. 
#### Running run_finbert.py(./src/finbert)
* Make sure to import AutoModelForSequenceClassification from transformers. It differs from the Prosus repo's old example
 since they just started migrating to transformers. Then again make sure you have the correct directory for the model.
 To calculate a set of sentiment-related scores for each piece of news, we did it like this:
     ```sh
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
            if summarize:
                text = text_summarization.summarize(text)
            finbert_score = predict(text=text, model=model, write_to_csv=False, path=None)
            print(finbert_score)
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
     ```
#### The finBERT predictions
* The prediction contains the corresponding sentence, the logit(positive, neutral or negative), prediction and sentiment_score.
We extract the logits for each category and we extract the score. the final finbert.json file is in the following format. 
Each company may have multiple articles so we take the average of the scores
for each company on each day.
  ```sh
    {
        "2016-01-28": {
            "FB": [
                {
                    "finbert": {
                        "negative": FLOAT,
                        "neutral": FLOAT,
                        "positive": FLOAT,
                        "sentiment_score": FLOAT
                    }
                }
            ]
        }
    }
  ```
#### Forming the LSTM training-ready data
```shell script
def finberData(summarize = False):
    filename = './json_files/finbert_with_summarize' if summarize else './json_files/finbert'
    with open(filename + '.json') as f:
        finbertJSON = json.load(f)
    df = pd.read_csv('./combined_prices_rl.csv', names=header)
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
        df[str(i) + 'finbert_sentiment_score'] = df.apply(lambda x: getFinData(x,date, 'sentiment_score'), axis=1)
        df = df.drop(date, axis=1)
    df = df.drop('day 31 date', axis=1)
    df = df.drop('company', axis=1)
    df = df.sample(frac=1, random_state=0)
    df.to_pickle(filename + '.pkl')

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

for method_name in methods:  #Goes over all the methods and produce the LSTM data.
    df = pd.read_pickle(method_name + '.pkl')  # 'finbert.pkl' in this case
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
```
#### finBERT Training:
* We used **LSTM** to train the model and the input to LSTM looks like this:
  ```sh
  [
    #data point 1
    [
      [closing_price, finbert_sentiment_score] # time step 1
      [closing_price, finbert_sentiment_score] # time step 2
      ...
      [closing_price, finbert_sentiment_score] # time step 30
    ]
    
    #data point 2
    [
      [closing_price, finbert_sentiment_score] # time step 1
      [closing_price, finbert_sentiment_score] # time step 2
      ...
      [closing_price, finbert_sentiment_score] # time step 30
    ]
    ...
  ]  
  ```
* LSTM Architecture:
  ```sh
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape = (dim[1], dim[2])))  # input_shape = (30, 2) in this case
  model.add(LSTM(50, return_sequences = False))
  model.add(Dense(25))
  model.add(Dense(1)) # 1 output: Price
  ```
* Training:
  * Use the adam optimizer.
  * Record train_loss and test_loss each epoch.
  * Use an EarlyStopper.
  * When training the model of the other methods, we used this same code, so we can compare them. We choose not to show this code again for the other methods, since they are the same.
  ```sh
  train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
  test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(model.evaluate(test_x, test_y)[0]))
  earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
  model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
  model.fit(train_x, train_y, batch_size=2000, epochs=epochs, callbacks=[train_loss, test_loss])
  ```
#### finBERT result:
* Plot the curve recorded by the the lambda function above and show the testing result:
   ```sh
   result = model.evaluate(test_x,test_y)[1]
   
   plt.figure()
   plt.title("Testing RMSE: " + str(result))
   plt.grid()
   plt.suptitle(method_name + " Learning Curve")
   plt.ylabel("loss")
   plt.xlabel("epochs")
   plt.ylim(top=max(train_scores),bottom=min(train_scores))
   plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, linewidth=1, color="r",
         label="Training loss")
   plt.plot(np.linspace(0,len(test_scores),len(test_scores)), test_scores, linewidth=1, color="b",
          label="Testing loss")
   legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
   legend.get_frame().set_facecolor('C0')

   plt.show()
   ```
![finBERT](./src/img/finBERT.png)
## [Sentence Transformers](https://www.sbert.net/)
* We mainly used the sentence_transformers, along with [LexRank](https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)
 for summarizing the news articles, as shown on their [repo](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/text-summarization/text-summarization.py).
 Since the articles can contain noise and finbert can be quite slow, we rank each article's content by centrality_scores
  in the following way and we take the top five sentences:
    ```sh
     #Compute the sentence embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)
  
    #Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
  
    #Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
  
    #We argsort so that the first element is the sentence with the highest score
    most_central_sentence_indices = np.argsort(-centrality_scores)
    summarization = ""
    for idx in most_central_sentence_indices[0:5]:
        summarization += sentences[idx]
    return summarization
    ```
* Then we use finBERT on the summarized articles to get the the sentiment score and trained another model.
* Sentence Transformers + finBERT result:
![Sentence Transformers + finBERT](./src/img/finbert_with_summarize.png)
## [NLTK Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
* Vader is one of NLTK's sentiment analysis library and it is quite straight forward to use. It contains 4 scores, negative,
neutral, positive and compound. We simply store the scores in the same fashion as we did for finbert.
    ````
        {
            "2016-01-28": {
                "FB": [
                    {
                        "vader": {
                            "compound": FLOAT,
                            "neg": FLOAT,
                            "neu": FLOAT,
                            "pos": FLOAT
                        }
                    }
                ]
            }
        }
    ````
* The code used to generate this file is ./src/vader/vader.py:
   ```sh
   nltk.download('vader_lexicon')
   sid = SentimentIntensityAnalyzer()

   f = open('./json_files/news.json')
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


   with open('./json_files/date_to_company_to_vader.json', 'w') as fp:
       json.dump(date_to_company_to_arrayOfMethodsScores, fp, sort_keys=True, indent=4)
   ```
* NLTK Vader result:
![vader](./src/img/vader.png)

## Goel and Mittal's Paper (Mood Analysis)
#### There are some problems to replicate this paper on our own dataset:
* 1. In the paper, the authors used tweets to analysis public mood, but in our case, the text data are news where it
  could be more difficult to extract than from the tweets since the tone in the news is more official and less casual.
* 2. The number of the news articles (29630 articles across 5 years) is very limited compare to the tweets the authors
  obtained (476 million across 6 months). There is a big chance that there won't be any mood extracted from a given
  day using the authors' method.
  
#### So, we made some adjustments to make this method at least usable on our dataset:
* 1. The author used SentiWordNet and a standard Thesaurus to find the synonyms of the POMS questionnaire and extend their
  word list by adding the synonyms. We chose to use **nltk.wordnet** and the word vectors model downloaded from fasttext.cc (**wiki-news-300d-1M.vec.zip**) 
  to extend our word list. Hence, the word list can be bigger so that we can have more matches when calculating the score for a POMS word.
* 2. we combined the news from all companies on a given day to find the moods instead of from individual companies. In this way, we will
  have more texts to extract moods from.
* 3. We find that the word list is still too small, so we used the word vector from fasttext.cc to further extend the word list. For more detail, please   look at the **fasttext section**.
  
#### Steps (./src/mittal_paper/):
* 1. We find that the POMS mentioned in the paper has evolved from the 65-word questionnaire to a 34-word questionnaire. 
  We then generated the synonyms of the words in the 34-word questionnaire by using the nltk wordnet as following:
  ``` sh
  # this dictionary is truncated, the full dictionary can be viewed in the file
  POMS_34_words_to_cat = {
    "tense": "ANX",
    "Angry": "ANG",
    "worn-out": "FAT",
    ...
  }
  syn_to_POMS = {}
  for word in POMS_34_words_to_cat:
    for ss in wordnet.synsets(word):
        for name in ss.lemma_names():
            if name not in syn_to_POMS:
                syn_to_POMS[name] = word

   with open('./json_files/syn_to_POMS_wordnet.json', 'w') as fp:
       json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)
  ```
* 2. We then calculate the score of each POMS state as mentioned in the paper: Score of a POMS word equals to the # of times the word matched in
  a day divided by # of total matches of all words. We then save a json file that maps a date to the moods score.
  Note that the "Angry" and "Anxious" states rarely appears in the new articles, so we decided to use only "Fatigue", "Vigorous", "Confusion" and   "Depression" as our moods.
  ```sh
  f1 = open('./json_files/syn_to_POMS_wordnet.json')
  SYN_TO_POMS = json.load(f1)
  def mittal_text_to_mood(articles):
    # this dictionary is truncated, the full dictionary can be viewed in the file
    POMS_34_words_score = {
                            "tense": 0,
                            "Angry": 0,
                            "worn-out": 0,
                            ...
    }
    for article in articles:
        for word in nltk.word_tokenize(article["text"]):
            if word.lower() in SYN_TO_POMS:
                POMS_34_words_score[SYN_TO_POMS[word.lower()]] += 1
    all_occurrance = sum(POMS_34_words_score.values())
    if all_occurrance == 0: return [0, 0, 0, 0]

    for word in POMS_34_words_score:
        POMS_34_words_score[word] /= all_occurrance 
    
    return poms_to_states(POMS_34_words_score)

  def poms_to_states(POMS_34_words_score):
      mood_states = {"ANX":0, "ANG":0, "FAT":0, "DEP":0, "VIG":0, "CON":0}
      for word in POMS_34_words_score:
         score = POMS_34_words_score[word]
         if score == 0: continue
         mood = POMS_34_words_to_cat[word]
         mood_states[mood] += math.ceil(score * 4)
      return list(mood_states.values())[2:]
      
   # date_to_articles_array.json is a dictionary with date as key and articles as value
   # The code used to generate the json file is ./src/mittal_paper/create_date_to_articles_array.py
   f2 = open('./json_files/date_to_articles_array.json')
   news = json.load(f2)

   date_to_moods = {}
   for date in news:
       date_to_moods[date] =  mittal_text_to_mood(news[date])

   with open('./json_files/date_to_moods.json', 'w') as fp:
       json.dump(date_to_moods, fp, sort_keys=True, indent=4)
  ```
  
* Goel and Mittal's Method result:
![mood](./src/img/mood.png)
## fasttext
#### We downloaded the word vectors wiki-news-300d-1M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html and used it to extend the word list that mentioned in the method in Mettal's Paper. 
* We simply added similar words of the POMS word into our word list. Although they are not entirely synonym, the result word list makes sense when we look at it. The code used to extend the word list is in ./src/fasttext/syn_to_POMS_fasttext.py
```sh

# Model downloaded from here: https://fasttext.cc/docs/en/english-vectors.html
model = gensim.models.KeyedVectors.load_word2vec_format('./src/fasttext/wiki-news-300d-1M.vec')

# this dictionary is truncated, the full dictionary can be viewed in the file
POMS_34_words_to_cat = {
    "tense": "ANX",
    "Angry": "ANG",
    "worn-out": "FAT",
    ...
 }
syn_to_POMS = {}
for word in POMS_34_words_to_cat:
    for similar_tup in model.most_similar(positive=[word],topn=10):
        similar = similar_tup[0].lower()
        if similar not in syn_to_POMS:
            syn_to_POMS[similar] = word

with open('./json_files/syn_to_POMS_fasttext.json', 'w') as fp:
    json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)
```
* Then we extend the wordlist generated by nltk wordnet as following: (./src/mittal_paper/combine_wordnet_fasttext.py)
```sh
f1 = open('./json_files/syn_to_POMS_wordnet.json')
SYN_TO_POMS = json.load(f1)
f2 = open('./json_files/syn_to_POMS_fasttext.json')
SYN_TO_POMS2 = json.load(f2)


for word in SYN_TO_POMS2:
    if word not in SYN_TO_POMS:
        SYN_TO_POMS[word] = SYN_TO_POMS2[word]

print(SYN_TO_POMS)
with open('./json_files/syn_to_POMS_combined.json', 'w') as fp:
    json.dump(SYN_TO_POMS, fp, sort_keys=True, indent=4)
```


## Software Repository for Accounting and Finance
* We chose to use "Loughran-McDonald Sentiment Word Lists" downloaded from https://sraf.nd.edu/textual-analysis/resources/. This table has seven **LM Sentiments**: Negative, Positive, Uncertainty, Litigious, Strong Modal, Weak Modal and Constraining. We use the same method mentioned in Goel Mittal's paper calculated each sentiment's score for each day. Again, we stored this feature as the same format as above methodes. We wrote two files to create this feature: **create_LM_Dictionary.py** and **sraf_sentiment.py**

* Create a dictionary that maps the LM sentiments to their words. 
```sh
file_name = "LoughranMcDonald_SentimentWordLists_2018.xlsx"
xl_file = pd.ExcelFile(file_name)

dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}


categories = ["Negative", "Positive", "Uncertainty", "Litigious", "StrongModal", "WeakModal", "Constraining"]

LM_Dict = {}
for cat in categories:
    first_word = dfs[cat].columns.values[0].lower()
    LM_Dict[cat] = {}
    LM_Dict[cat][first_word] = 0
    for word in dfs[cat][first_word.upper()]:
        LM_Dict[cat][word.lower()] = 0

with open('./json_files/LM_Dict.json', 'w') as fp:
    json.dump(LM_Dict, fp, sort_keys=True, indent=4)
    
f = open('./json_files/LM_Dict.json')
LM_Dict = json.load(f)

```
* Calculate the LM sentiment score and save them as a file:

```sh
def LM_text_to_sentiment():
    f = open('./json_files/news.json')
    date_to_company_to_sraf = {}
    data = json.load(f)
    for company in data:
	articles_for_company = data[company]
	for article in articles_for_company:
            time = article["pub_time"][0:10]
	    score = calculate_sraf(article["text"])
       	    if time in date_to_company_to_sraf:
		if company in date_to_company_to_sraf[time]:
		    date_to_company_to_sraf[time][company].append({"sraf": score})
		else:
		    date_to_company_to_sraf[time][company] = [{"sraf": score}]
	    else:
		date_to_company_to_sraf[time] = {}
		date_to_company_to_sraf[time][company] = [{"sraf":score}]
	return date_to_company_to_sraf

def calculate_sraf(text):
    sentiments = {"Negative":0, "Positive":0, "Uncertainty":0, "Litigious":0, "StrongModal":0, "WeakModal":0, "Constraining":0}
    for word in nltk.word_tokenize(text):
	for sentiment in sentiments:
	    if word.lower() in LM_Dict[sentiment]:
		sentiments[sentiment] += 1
		    break

    all_occurrance = sum(sentiments.values())
    if all_occurrance == 0: return [0, 0, 0, 0, 0, 0, 0]
    for sentiment in sentiments:
	sentiments[sentiment] /= all_occurrance
    return list(sentiments.values())


with open('./json_files/date_to_company_to_sraf.json', 'w') as fp:
    json.dump(LM_text_to_sentiment(), fp, sort_keys=True, indent=4)
```
* SRAF result:
![SRAF](./src/img/sraf.png)

## Ensemble
* We used all features generated by the above method and trained a final model. The result is shown below.
* in Summary, our model is ensembled in the following way:
  * we used **sentence transformer** to summarize the articles, then used **finBERT** to extract sentiment from the articles.
  * we used **NLTK Vader** to give polarity score of the articles, this is similar to the step above.
  * we used the **mood analysis strategy in Goel and Mittal's paper** to extract moods from the articles of a given day.
  * we used **SRAF** to extract LM Sentiment of the articles.
* result:
![ALL](./src/img/alldata.png)

## More Experiements:
* We also tried something different with the ensembled model
  * Since the result using **SRAF** only is not ideal compared to other methods, we removed this feature.
  * Since **NLTK Vader** and **finBERT** have similar purpose, i.e, polarity, we removed the NLTK Vader Score and kept the feature generated by sentence transformer + finBERT.
* And we kept the result of the features and trained a model, but the result is not as good as the above ensembled model which contains all the features.

* result:
![all_data_novader_nosraf](./src/img/all_data_novader_nosraf.png)

## Possible Improvements:
* Gather more news data. Right now, they are to sparse compare to the prehistorical data we have.
* Gather some data from social media like tweeter which can be easier to extract public moods from.
* Extend the word list in mood analysis method (Goel and Mittal's paper).
* Try different LSTM hyper-parameter and tune them.
* Try use earlier data as input, for example, use the previous 60 days data instead of 30. 
* Try a different method when using the SRAF LM sentiments.
