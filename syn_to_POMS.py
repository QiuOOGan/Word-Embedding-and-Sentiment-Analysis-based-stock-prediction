import json
from nltk.corpus import wordnet

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

syn_to_POMS = {}
for word in POMS_65_words:
    for ss in wordnet.synsets(word):
        for name in ss.lemma_names():
            if name not in syn_to_POMS:
                syn_to_POMS[name] = word

with open('syn_to_POMS.json', 'w') as fp:
    json.dump(syn_to_POMS, fp, sort_keys=True, indent=4)