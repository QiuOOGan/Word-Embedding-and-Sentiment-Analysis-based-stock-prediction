import json
import nltk
import math

f = open('LM_Dict.json')
LM_Dict = json.load(f)



def LM_text_to_sentiment():
	f = open('news.json')
	counter = 0
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
			counter+=1
			print("done: ", counter)
	return date_to_company_to_sraf

				




def calculate_sraf(text):
	sentiments = {"Negative":0, "Positive":0, "Uncertainty":0, "Litigious":0, "StrongModal":0, "WeakModal":0, "Constraining":0}
	for word in nltk.word_tokenize(text):
		for sentiment in sentiments:
			if word.lower() in LM_Dict[sentiment]:
				sentiments[sentiment] += 1
				break

	all_occurrance = sum(sentiments.values())
	if all_occurrance == 0: return [0, 0, 0, 0, 0, 0]
	for sentiment in sentiments:
		sentiments[sentiment] /= all_occurrance
	return list(sentiments.values())


with open('date_to_company_to_sraf.json', 'w') as fp:
    json.dump(LM_text_to_sentiment(), fp, sort_keys=True, indent=4)

