# Word Embedding and Sentiment Analysis Based Stock Prediction

In this project, we used several word embedding methods on the financial news dataset to create additional features to the prehistorical stock price data set and trained a model to predict the close price at a specific time. The main tools we used are based on a language model called "BERT", i.e Bidirectional Encoder Representations from Transformers. In this project, we will train the model with each of the tools listed below and compare their results. In the end, we will try to ensemble the tools to train a final model.

## Tools:
* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
    * [finBERT](https://github.com/ProsusAI/finBERT)
    * [Sentence Transfermers](https://github.com/UKPLab/sentence-transformers)
* [fasttext](https://fasttext.cc/)
* [NLTK Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
* [GoelMittal's Paper](http://cs229.stanford.edu/proj2011/GoelMittal-StockMarketPredictionUsingTwitterSentimentAnalysis.pdf)
* [Software Repository for Accounting and Finance](https://sraf.nd.edu/textual-analysis/)

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
    
## Prerequisites

## finBERT

## Sentence Transfermers

## NLTK Vader

## GoelMittal's Paper (Mood Analysis)
#### There are some problems to replicate this paper on our own dataset:
* 1. In the paper, the authors used tweets to analysis public mood, but in our case, the text data are news where it
  could be more difficult to extract than from the tweets since the tone in the news is more official and less casual.
* 2. The number of the news articles (29630 articles across 5 years) is very limited compare to the tweets the authors
  obtained (476 million across 6 months). There is a big chance that there won't be any mood extracted from a given
  day using the authors' method.
  
#### So, we made some adjustments to make this method at least usable on our dataset:
* 1. The author used SentiWordNet and a standard Thesaurus to find the synonyms of the POMS questionnaire and extend their
  word list by adding the synonyms. We chose to use nltk.wordnet and the word vectors model downloaded from fasttext.cc (wiki-news-300d-1M.vec.zip) 
  to extend our word list. Hence, the word list can be bigger so that we can have more matches when calculating the score for a POMS word.
* 2. we combined the news from all companies on a given day to find the moods instead of from individual companies. In this way, we will
  have more texts to extract moods from.
* 3. We find that the word list is still too small, so we used the word vector from fasttext.cc to further extend the word list. For more detial, please   look at the fasttext section.
  
#### Steps (./src/mittal_paper/):
* 1. We find that the POMS mentioned in the paper has evolved from the 65-word questionnaire to a 34-word questionnaire. 
  We then generated the synonyms of the words in the 34-word questionnaire by using the nltk wordnet as following:
  ``` sh
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

   with open('syn_to_POMS_wordnet.json', 'w') as fp:
       json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)
  ```
* 2. We then calculate the score of each POMS state as mentioned in the paper: Score of a POMS word equals to the # of times the word matched in
  a day divided by # of total matches of all words. We then save a json file that maps a date to the moods score.
  Note that the "Angry" and "Anxious" states rarely appears in the new articles, so we decided to use only "Fatigue", "Vigorous", "Confusion" and   "Depression" as our moods.
  ```sh
  f1 = open('syn_to_POMS_wordnet.json')
  SYN_TO_POMS = json.load(f1)
  def mittal_text_to_mood(articles):
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
   f2 = open('date_to_articles_array.json')
   news = json.load(f2)

   date_to_moods = {}
   for date in news:
       date_to_moods[date] =  mittal_text_to_mood(news[date])

   with open('date_to_moods.json', 'w') as fp:
       json.dump(date_to_moods, fp, sort_keys=True, indent=4)
  ```
## fasttext
#### We downloaded the word vectors wiki-news-300d-1M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html and used it to extend the word list that mentioned in the method in Mettal's Paper. 
* We simply added similar words of the POMS word into our word list. Although they are not entirely synonym, the result word list makes sense when we look at it. The code used to extend the word list is in ./src/fasttext/syn_to_POMS_fasttext.py
```sh

# Model downloaded from here: https://fasttext.cc/docs/en/english-vectors.html
model = gensim.models.KeyedVectors.load_word2vec_format('./src/fasttext/wiki-news-300d-1M.vec')
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

with open('syn_to_POMS_fasttext.json', 'w') as fp:
    json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)
```



## Software Repository for Accounting and Finance
