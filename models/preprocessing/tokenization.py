from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")


def tokenize(s):
    return tokenizer(s, return_tensors="pt", padding=True)
