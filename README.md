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

## Data set:
* news.json:
it consists news articles from 81 big companies. Each company has an array of articles with field of "title", "text", "pul_time", and "url". Here is an example of "FaceBook":
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
    They contain minute-level historical prices in the past 5 years for 86 companies. Here is how they look like:
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

## fasttext

## NLTK Vader

## GoelMittal's Paper

## Software Repository for Accounting and Finance
