from glob import glob
import json
from tqdm import tqdm
jsonList = []
for path in glob("./jawiki/*/wiki_*", recursive=True):
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            jsonList.append(json.loads(line))
valid_paragraphs = []
for art in tqdm(jsonList):
    text = art["text"].split('\n')
    for paragraph in text:
        sentences = [s for s in paragraph.split('。') if len(s) > 0]
        if len(sentences) > 1:
            valid_paragraphs.append(sentences)

with open("./jawiki/japanese_wiki_paragraphs.json", 'w') as f:
    json.dump(valid_paragraphs, f, indent=2)