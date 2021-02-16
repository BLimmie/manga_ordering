import os.path as op
from typing import List

from data.constants import root


def get_books() -> List[str]:
    with open(op.join(root, "manga109", "books.txt")) as f:
        books = f.readlines()
        return [book.strip() for book in books]
