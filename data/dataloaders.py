from torch.utils.data import DataLoader, Dataset
import json
class WikiLoader(Dataset):
    def __init__(self, wikijsonpath):
        with open(wikijsonpath) as f:
            self.data = json.load(f)

    def __getitem__(self, item):
        pass
