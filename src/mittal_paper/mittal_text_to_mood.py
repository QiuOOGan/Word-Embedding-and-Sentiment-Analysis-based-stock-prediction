import json
import nltk
import math

f1 = open('syn_to_POMS_combined.json')
SYN_TO_POMS = json.load(f1)



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

def mittal_text_to_mood(articles):
    POMS_34_words_score = {
                            "tense": 0,
                            "Angry": 0,
                            "worn-out": 0,
                            "unhappy": 0,
                            "lively": 0,
                            "confused": 0,
                            "sad": 0,
                            "active": 0,
                            "on edge": 0,
                            "grumpy": 0,
                            "energetic": 0,
                            "hopeless": 0,
                            "uneasy": 0,
                            "restless": 0,
                            "distracted": 0,
                            "fatigued": 0,
                            "annoyed": 0,
                            "discouraged": 0,
                            "resentful": 0,
                            "nervous": 0,
                            "miserable": 0,
                            "bitter": 0,
                            "exhausted": 0,
                            "anxious": 0,
                            "helpless": 0,
                            "weary": 0,
                            "energized": 0,
                            "bewildered": 0,
                            "furious": 0,
                            "worthless": 0,
                            "forgetful": 0,
                            "vigorous": 0,
                            "uncertain": 0,
                            "drained": 0
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
    # return POMS_34_words_score


def poms_to_states(POMS_34_words_score):
    mood_states = {"ANX":0, "ANG":0, "FAT":0, "DEP":0, "VIG":0, "CON":0}
    for word in POMS_34_words_score:
        score = POMS_34_words_score[word]
        if score == 0: continue
        mood = POMS_34_words_to_cat[word]
        mood_states[mood] += math.ceil(score * 4)

    return list(mood_states.values())[2:]

#TODO: How to calculate calm, alert and kind
# def states_to_final_4_mood(mood_states):
#     calm, happy, alert, kind = 0, 0, 0, 0
#     happy = mood_states["VIG"] + (-mood_states["DEP"])
        



# Usage:
f2 = open('date_to_articles_array.json')
news = json.load(f2)

date_to_moods = {}
for date in news:
    date_to_moods[date] =  mittal_text_to_mood(news[date])

with open('date_to_moods.json', 'w') as fp:
    json.dump(date_to_moods, fp, sort_keys=True, indent=4)


