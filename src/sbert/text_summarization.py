"""
This example uses LexRank (https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)
to create an extractive summarization of a long document.

The document is splitted into sentences using NLTK, then the sentence embeddings are computed. We
then compute the cosine-similarity across all possible sentence pairs.

We then use LexRank to find the most central sentences in the document, which form our summary.

Input document: First section from the English Wikipedia Section
Output summary:
Located at the southern tip of the U.S. state of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban landmass.
New York City (NYC), often called simply New York, is the most populous city in the United States.
Anchored by Wall Street in the Financial District of Lower Manhattan, New York City has been called both the world's leading financial center and the most financially powerful city in the world, and is home to the world's two largest stock exchanges by total market capitalization, the New York Stock Exchange and NASDAQ.
New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce, entertainment, research, technology, education, politics, tourism, art, fashion, and sports.
If the New York metropolitan area were a sovereign state, it would have the eighth-largest economy in the world.
"""
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.sbert.LexRank import degree_centrality_scores



model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Our input document we want to summarize
# As example, we take the first section from Wikipedia
def summarize(document):
    # document = \
        # "NEW YORK, Aug 16 (Reuters) - U.S. stocks rebounded on Friday as an ebbing bond rally and news of potential German economic stimulus brought buyers back to equities, but major indexes still ended the week with losses.\n\nBased on the latest available data, the Dow Jones Industrial Average rose 307.69 points, or 1.2%, to 25,887.08, the S&P 500 gained 41.23 points, or 1.45%, to 2,888.83 and the Nasdaq Composite added 129.38 points, or 1.67%, to 7,895.99. (Reporting by Stephen Culp; editing by Jonathan Oatis)"
    #Split the document into sentences
    sentences = nltk.sent_tokenize(document)
    print("Num sentences:", len(sentences))

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


#Print the 5 sentences with the highest scores
# print("\n\nSummary:")
# for idx in most_central_sentence_indices[0:5]:
#     print(sentences[idx].strip())
