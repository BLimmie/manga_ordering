from typing import List, Tuple, Optional

from data.annotations import annotations
from data.books import get_books
from data.constants import degeneracies, validation, testing


def get_pages_partition(split="all", *args, **kwargs):
    if split == "train":
        exclude = validation + testing
    else:
        exclude = None
    if split == "validation":
        include = validation
    elif split == "test":
        include = testing
    elif split == "smoke":
        include = ["YouchienBoueigumi"]
    elif split == "all" or split == "train":
        include = None
    else:
        raise TypeError("invalid split type")
    return get_pages(*args, exclude=exclude, include=include, **kwargs)


def get_pages(with_text: bool = False, exclude: Optional[List[str]] = None, include: Optional[List[str]] = None) -> \
        List[Tuple[str, int]]:
    assert exclude is None or include is None
    if exclude is None:
        exclude = []
    result = []
    books = get_books() if include is None else include
    for book in books:
        if book in exclude:
            continue
        data = annotations(book)
        num_pages = len(data["book"]["pages"]["page"])
        for i in range(num_pages):
            if with_text:
                if "text" not in data["book"]["pages"]["page"][i]:
                    continue
                if (book, i) in degeneracies:
                    continue
            result.append((book, i))
    return result
