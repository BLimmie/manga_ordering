from typing import List, Tuple

from data.annotations import annotations
from data.books import get_books
from data.constants import degeneracies


def get_pages(with_text=False) -> List[Tuple[str, int]]:
    result = []
    books = get_books()
    for book in books:
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
