import json
import gensim

# Not Using This Questionaire since I can't find the map to the six states mood
POMS_65_words = ["friendly", "tense", "angry", "worn out",
                "unhappy", "clear headed", "lively", "confused",
                "sorry for things done", "shaky", "listless", "peeved",
                "considerate", "sad", "active", "on edge",
                "grouchy", "blue", "energetic",
                "panicky", "hopeless", "relaxed", "unworthy",
                "spiteful", "sympathetic", "uneasy", "restless",
                "unable to concentrate", "fatigued", "helpful",
                "annoyed", "discouraged", "resentful", "nervous",
                "lonely", "miserable", "muddled", "cheerful", "bitter",
                "exhausted", "anxious", "ready to fight", "good natured",
                "gloomy", "desperate", "sluggish", "rebellious",
                "helpless", "weary", "bewildered", "alert", "deceived",
                "furious", "efficient", "trusting", "full of pep", "bad tempered",
                "worthless", "forgetful", "carefree", "terrified", "guilty",
                "vigorous", "uncertain about things", "bushed"]

# https://soulandspiritmagazine.com/wp-content/uploads/2018/08/Forest-bathing-download.pdf
POMS_34_words_to_cat = {
    "tense": "ANX",
    "Angry": "ANG",
    "worn-out": "FAT",
    "unhappy": "DEP",
    "lively": "VIG",
    "confused": "CON",
    "sad": "DEP",
    "active": "VIG",
    "on-edge": "ANX",
    "grumpy": "ANG",
    "energetic": "VIG",
    "hopeless": "DEP",
    "uneasy": "ANX",
    "restless": "ANX",
    "distracted": "CON",
    "fatigued": "FAT",
    "annoyed": "ANG",
    "discouraged": "DEP",
    "resentful": "ANG",
    "nervous": "ANX",
    "miserable": "DEP",
    "bitter": "ANG",
    "exhausted": "FAT",
    "anxious": "ANX",
    "helpless": "DEP",
    "weary": "FAT",
    "energized": "VIG",
    "bewildered": "CON",
    "furious": "VIG",
    "worthless": "DEP",
    "forgetful": "CON",
    "vigorous": "VIG",
    "uncertain": "CON",
    "drained": "FAT"
}

for key in POMS_34_words_to_cat:
    print(key, POMS_34_words_to_cat[key])


model = gensim.models.KeyedVectors.load_word2vec_format('./src/fasttext/wiki-news-300d-1M.vec')

syn_to_POMS = {}
for word in POMS_34_words_to_cat:
    for similar_tup in model.most_similar(positive=[word],topn=10):
        similar = similar_tup[0].lower()
        if similar not in syn_to_POMS:
            syn_to_POMS[similar] = word

with open('./json_files/syn_to_POMS_fasttext.json', 'w') as fp:
    json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)
