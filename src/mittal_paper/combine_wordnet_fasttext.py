import json

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



