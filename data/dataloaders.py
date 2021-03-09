import json
import os.path as op
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
from tqdm import trange
from transformers import BertTokenizer

from .pages import get_pages_partition


def singleton_collate_fn(data):
    return data[0]


def scores(x, shuffle=True):
    step = 1.0 / (len(x) - 1)
    y = [i * step for i in range(len(x))]
    z = list(zip(x, y))
    if shuffle:
        random.shuffle(z)
    x, y = zip(*z)
    return list(x), torch.Tensor(y)


class WikiData(Dataset):
    def __init__(self, wikijsonpath, offset=0, divisions=5, test=False):
        self.offset = offset
        self.divisions = divisions
        with open(wikijsonpath) as f:
            self.data = json.load(f)
        self.test = test

    def __getitem__(self, index):
        x = self.data[index * self.divisions + self.offset]
        return scores(x, not self.test)

    def __len__(self):
        return len(self.data) // self.divisions - 2


class PairwiseWikiData(Dataset):
    def __init__(self, wikijsonpath, offset=0, divisions=20, test=False):
        self.test = test
        self.tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        first = []
        second = []
        with open(wikijsonpath) as f:
            data = json.load(f)
        for i in trange(offset, len(data), divisions):
            sentences = data[i]
            for j in range(len(sentences)):
                for k in range(j + 1, len(sentences)):
                    sent1, sent2 = sentences[j], sentences[k]
                    s1i = self.tokenizer.encode(sent1)
                    s2i = self.tokenizer.encode(sent2)
                    if len(s1i) < 50:
                        sent2 = self.tokenizer.decode(s2i[:100 - len(s1i)])
                    elif len(s2i) < 50:
                        sent1 = self.tokenizer.decode(s1i[:100 - len(s2i)])
                    else:
                        sent1 = self.tokenizer.decode(s1i[:50])
                        sent2 = self.tokenizer.decode(s2i[:50])
                    first.append(sent1)
                    second.append(sent2)
        self.data = self.tokenizer(first + second, second + first, add_special_tokens=True, padding=True,
                                   return_tensors="pt")
        self.labels = ([0] * (len(first)) + [1] * (len(second)))

    def __getitem__(self, idx):
        encodings = {key: val[idx].clone().detach() for key, val in self.data.items()}
        if not self.test:
            encodings['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encodings

    def __len__(self):
        return len(self.labels)


class SentenceDataset(Dataset):
    def __init__(self, sentences):

        self.tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        first = []
        second = []
        self.pos = []

        for j in range(len(sentences)):
            for k in range(j + 1, len(sentences)):
                sent1, sent2 = sentences[j], sentences[k]
                s1i = self.tokenizer.encode(sent1)
                s2i = self.tokenizer.encode(sent2)
                if len(s1i) < 50:
                    sent2 = self.tokenizer.decode(s2i[:100 - len(s1i)])
                elif len(s2i) < 50:
                    sent1 = self.tokenizer.decode(s1i[:100 - len(s2i)])
                else:
                    sent1 = self.tokenizer.decode(s1i[:50])
                    sent2 = self.tokenizer.decode(s2i[:50])
                first.append(sent1)
                second.append(sent2)
                self.pos.append((j, k))
        self.data = self.tokenizer(first, second, add_special_tokens=True, padding=True,
                                   return_tensors="pt")

    def __getitem__(self, idx):
        encodings = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return encodings

    def __len__(self):
        return len(self.data["input_ids"])


class PageDataset(SentenceDataset):
    def __init__(self, data, imagepath, page):
        self.page = page
        self.imagepath = imagepath
        self.transform = Resize((1170, 1654))
        sentences = data[page[0]][str(page[1])]
        super().__init__(sentences)

    def __getitem__(self, idx):
        encodings = super().__getitem__(idx)
        img = read_image(op.join(self.imagepath, self.page[0], f"{self.page[1]:03d}.jpg")).float().div(255)
        if img.shape != torch.Size([3, 1170, 1654]):
            img = self.transform(img)
        encodings["img"] = img
        return encodings

    def __len__(self):
        return len(self.data["input_ids"])


def WikiLoader(jsonpath, offset=0, divisions=5, batch_size=1, test=False, *args, **kwargs) -> DataLoader:
    assert batch_size == 1, "The models only support batch sizes of 1"
    return DataLoader(WikiData(jsonpath, offset, divisions, test), batch_size=1, collate_fn=singleton_collate_fn, *args,
                      **kwargs)


def WikiTopologicalLoader(jsonpath, offset=0, divisions=20, batch_size=128, *args, **kwargs) -> DataLoader:
    return DataLoader(PairwiseWikiData(jsonpath, offset, divisions), batch_size=batch_size, *args, **kwargs)


class MangaDataNoIMG(Dataset):
    def __init__(self, jsonpath, split="all"):
        with open(jsonpath) as f:
            self.data = json.load(f)
        self.pages = get_pages_partition(with_text=True, split=split)
        self.test = split == "test"

    def __getitem__(self, index):
        page = self.pages[index]
        x = self.data[page[0]][str(page[1])]
        if len(x) == 1:
            return None, None, None
        return *scores(x, not self.test), None

    def __len__(self):
        return len(self.pages)


class MangaData(MangaDataNoIMG):
    def __init__(self, jsonpath, imagepath, split="all"):
        assert imagepath is not None
        self.imagepath = imagepath
        self.transform = Resize((1170, 1654))
        super().__init__(jsonpath, split=split)

    def __getitem__(self, index) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, y, _ = super().__getitem__(index)
        if x is None or y is None:
            return None, None, None

        page = self.pages[index]

        img = read_image(op.join(self.imagepath, page[0], f"{page[1]:03d}.jpg")).float().div(255)
        if img.shape != torch.Size([3, 1170, 1654]):
            img = self.transform(img)
        img = img.unsqueeze(0)
        return x, y, img


class PairwiseMangaDataNoIMG(Dataset):
    def __init__(self, jsonpath, split="all"):
        self.pages = get_pages_partition(with_text=True, split=split)
        self.example_to_page = []
        self.test = split == "test"
        self.tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        first = []
        second = []
        with open(jsonpath) as f:
            data = json.load(f)
        for i in trange(len(self.pages)):
            page = self.pages[i]
            sentences = data[page[0]][str(page[1])]
            for j in range(len(sentences)):
                for k in range(j + 1, len(sentences)):
                    sent1, sent2 = sentences[j], sentences[k]
                    s1i = self.tokenizer.encode(sent1)
                    s2i = self.tokenizer.encode(sent2)
                    if len(s1i) < 50:
                        sent2 = self.tokenizer.decode(s2i[:100 - len(s1i)])
                    elif len(s2i) < 50:
                        sent1 = self.tokenizer.decode(s1i[:100 - len(s2i)])
                    else:
                        sent1 = self.tokenizer.decode(s1i[:50])
                        sent2 = self.tokenizer.decode(s2i[:50])
                    first.append(sent1)
                    second.append(sent2)
                    self.example_to_page.append(page)
        self.data = self.tokenizer(first + second, second + first, add_special_tokens=True, padding=True,
                                   return_tensors="pt")
        self.labels = ([0] * (len(first)) + [1] * (len(second)))
        self.example_to_page.extend(self.example_to_page)

    def __getitem__(self, idx):
        encodings = {key: val[idx].clone().detach() for key, val in self.data.items()}
        if not self.test:
            encodings['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encodings

    def __len__(self):
        return len(self.labels)


class PairwiseMangaData(PairwiseMangaDataNoIMG):
    def __init__(self, jsonpath, imagepath, split="all"):
        self.imagepath = imagepath
        self.transform = Resize((1170, 1654))
        super().__init__(jsonpath, split=split)

    def __getitem__(self, idx):
        encodings = super().__getitem__(idx)
        page = self.example_to_page[idx]
        img = read_image(op.join(self.imagepath, page[0], f"{page[1]:03d}.jpg")).float().div(255)

        if img.shape != torch.Size([3, 1170, 1654]):
            img = self.transform(img)
        return encodings, img

    def __len__(self):
        return len(self.labels)


def MangaLoader(jsonpath, imagepath=None, images=False, batch_size=1, split="all", *args, **kwargs) -> DataLoader:
    assert batch_size == 1, "The models only support batch sizes of 1"
    if not images:
        return DataLoader(MangaDataNoIMG(jsonpath, split=split), batch_size=1, collate_fn=singleton_collate_fn, *args,
                          **kwargs)
    else:
        return DataLoader(MangaData(jsonpath, imagepath, split=split), batch_size=1, collate_fn=singleton_collate_fn,
                          *args,
                          **kwargs)
