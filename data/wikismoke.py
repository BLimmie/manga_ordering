import json

with open("jawiki/japanese_wiki_paragraphs.json") as f:
    data = json.load(f)
    smokedata = data[8500:9000]

with open("jawiki/japanese_wiki_smoke.json", 'w') as f:
    json.dump(smokedata, f, indent=2)