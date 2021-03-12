import json
import pandas as pd


# Downloaded from here: https://sraf.nd.edu/textual-analysis/resources/#LM%20Sentiment%20Word%20Lists
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

with open('LM_Dict.json', 'w') as fp:
    json.dump(LM_Dict, fp, sort_keys=True, indent=4)
