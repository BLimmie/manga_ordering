import json
import os.path as op
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

from .pages import get_pages


def singleton_collate_fn(data):
    return data[0]


def scores(x):
    step = 1.0 / (len(x) - 1)
    y = [i * step for i in range(len(x))]
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    return list(x), torch.Tensor(y)


class WikiData(Dataset):
    def __init__(self, wikijsonpath, offset=0, divisions=5):
        self.offset = offset
        self.divisions = divisions
        with open(wikijsonpath) as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        x = self.data[index * self.divisions + self.offset]
        return scores(x)

    def __len__(self):
        return len(self.data) // self.divisions - 2


def WikiLoader(jsonpath, offset=0, divisions=10, batch_size=1, *args, **kwargs) -> DataLoader:
    assert batch_size == 1, "The models only support batch sizes of 1"
    return DataLoader(WikiData(jsonpath, offset, divisions), batch_size=1, collate_fn=singleton_collate_fn, *args,
                      **kwargs)


class MangaDataNoIMG(Dataset):
    def __init__(self, jsonpath):
        with open(jsonpath) as f:
            self.data = json.load(f)
        self.pages = get_pages(with_text=True)

    def __getitem__(self, index):
        page = self.pages[index]
        x = self.data[page[0]][str(page[1])]
        if len(x) == 1:
            return None, None
        return scores(x)

    def __len__(self):
        return len(self.pages)


class MangaData(MangaDataNoIMG):
    def __init__(self, jsonpath, imagepath):
        assert imagepath is not None
        self.imagepath = imagepath
        self.transform = Resize((1170, 1654))
        super().__init__(jsonpath)

    def __getitem__(self, index) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, y = super().__getitem__(index)
        if x is None or y is None:
            return None, None, None

        page = self.pages[index]

        img = read_image(op.join(self.imagepath, page[0], f"{page[1]:03d}.jpg")).float().div(255)
        if img.shape != torch.Size([3, 1170, 1654]):
            img = self.transform(img)

        return x, y, img


def MangaLoader(jsonpath, imagepath=None, images=False, batch_size=1, *args, **kwargs) -> DataLoader:
    assert batch_size == 1, "The models only support batch sizes of 1"
    if not images:
        return DataLoader(MangaDataNoIMG(jsonpath), batch_size=1, collate_fn=singleton_collate_fn, *args, **kwargs)
    else:
        return DataLoader(MangaData(jsonpath, imagepath), batch_size=1, collate_fn=singleton_collate_fn, *args,
                          **kwargs)
