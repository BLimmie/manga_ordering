import json
import os.path as op
from typing import Dict, Optional

import xmltodict

from data.books import get_books
from data.constants import root


class AnnotationsWrapper:
    def __init__(self):
        self.annotations = {}
        for book in get_books():
            with open(op.join(root, "manga109", "annotations_json", book + ".json"), encoding="utf-8") as f:
                self.annotations[book] = json.load(f)

    def __getitem__(self, item):
        return self.annotations[item]

    def page_annotations(self, title, page_no: int):
        return self[title]["book"]["pages"]["page"][page_no]


try:
    annotations_cache = AnnotationsWrapper()
except:
    print("Some functions may not work")


def annotations_xml(title: str) -> Dict:
    with open(op.join(root, "manga109", "annotations", title + ".xml"), encoding="utf-8") as f:
        data = xmltodict.parse(f.read())
    return data


def annotations(title: str, page_no: Optional[int] = None) -> Dict:
    if page_no is None:
        return annotations_cache[title]
    return annotations_cache.page_annotations(title, page_no)
