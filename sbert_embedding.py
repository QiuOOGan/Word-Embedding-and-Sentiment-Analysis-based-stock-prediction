from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Our sentences we like to encode
sentences = ["NEW YORK, Aug 16 (Reuters) - U.S. stocks rebounded on Friday as an ebbing bond rally and news of potential German economic stimulus brought buyers back to equities, but major indexes still ended the week with losses.\n\nBased on the latest available data, the Dow Jones Industrial Average rose 307.69 points, or 1.2%, to 25,887.08, the S&P 500 gained 41.23 points, or 1.45%, to 2,888.83 and the Nasdaq Composite added 129.38 points, or 1.67%, to 7,895.99. (Reporting by Stephen Culp; editing by Jonathan Oatis)",
    "Sept 19 (Reuters) - Wall Street ended mixed on Thursday, with a dip in Apple Inc shares offsetting a gain in Microsoft Corp shares a day after the Federal Reserve cut interest rates as expected and left the door open for further monetary easing.\n\nThe Dow Jones Industrial Average fell 52.9 points, or 0.19%, to 27,094.18, the S&P 500 gained 0.03 points, or 0.00%, to 3,006.76 and the Nasdaq Composite added 5.49 points, or 0.07%, to 8,182.88. (Reporting by Noel Randewich)",
    "(Reuters) - Visa Inc, Mastercard Inc, and a number of U.S. banks on Tuesday agreed to pay $6.2 billion to settle a long-running lawsuit brought by merchants over the fees they pay when they accept card payments.\n\nVisa and Mastercard previously reached a $7.25 billion settlement with the merchants in the case, but that deal was thrown out by a federal appeals court in 2016 and the U.S. Supreme Court last year refused to revive it.\n\nThe deal had been the largest all-cash U.S. antitrust settlement, although its value shrank to $5.7 billion after roughly 8,000 retailers opted out.\n\nThe card issuers named in the class-action lawsuit include JPMorgan Chase & Co, Citigroup and Bank of America.\n\nThe lawsuit, brought on behalf of about 12 million retailers and dating back more than a decade, accuses the credit card companies of violating federal antitrust laws by forcing merchants to pay swipe fees and prohibiting them from directing consumers toward other methods of payment.\n\nSlideshow ( 2 images )\n\nIn rejecting the earlier settlement, which was opposed by retailers including Amazon.com Inc, Costco Wholesale Corp and Walmart Inc, a federal appeals court found that the accord was unfair because some retailers would receive little or no benefit.\n\nThe card companies have already paid $5.3 billion and will now pay an additional $900 million.\n\nMastercard will pay an additional $108 million from funds set aside in the second quarter, the company said reut.rs/2OA2V0i.\n\nVisa\u2019s share represents around $4.1 billion, which the company expects to pay using funds previously deposited with the court, and from a litigation escrow it set up on June 28.\n\nThe settlement must still be approved by a court."]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("Embedding length:", len(embedding))
    print("")