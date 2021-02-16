import json
import torch
from torch.utils.data import DataLoader, Dataset
import random

def singleton_collate_fn(data):
    return data[0]

class WikiData(Dataset):
    def __init__(self, wikijsonpath):
        with open(wikijsonpath) as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        x = self.data[index]
        step = 1.0/(len(x)-1)
        y = [i*step for i in range(len(x))]
        z = list(zip(x,y))
        random.shuffle(z)
        x,y = zip(*z)

        return list(x), torch.Tensor(y)

    def __len__(self):
        return len(self.data)


def WikiLoader(jsonpath, batch_size=1, *args, **kwargs) -> DataLoader:
    assert batch_size == 1, "The models only support batch sizes of 1"
    return DataLoader(WikiData(jsonpath), batch_size=1, collate_fn=singleton_collate_fn, *args, **kwargs)

