from tqdm import tqdm
from data.data_processing import text_order
import json
from data.pages import get_pages

pages = get_pages(with_text=True)
data = {}
for page in tqdm(pages, total=len(pages)):
    if page[0] not in data:
        data[page[0]] = {}

    data[page[0]][page[1]] = text_order(*page)

with open("manga109/sentence_order.json", 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)