import os.path as op
from typing import Optional

import cv2
import numpy as np

from data.annotations import annotations
from data.constants import root, image_shape, image_shape_exception, size_title_exception


def frame_crosses_middle(title, page_no) -> Optional[bool]:
    data = annotations(title)
    img = cv2.imread(op.join(root, "manga109", "images", title, f"{page_no:03d}.jpg"))
    mid = img.shape[1] / 2
    if "frame" not in data["book"]["pages"]["page"][page_no]:
        return
    frames = data["book"]["pages"]["page"][page_no]["frame"]
    if not isinstance(frames, list):
        frames = [frames]
    for box in frames:
        x1 = int(box["@xmin"])
        x2 = int(box["@xmax"])
        if x1 < mid - 75 < mid + 75 < x2:
            return True
    return False


def find_midpoint(title, page_no) -> int:
    img = cv2.imread(op.join(root, "manga109", "images", title, f"{page_no:03d}.jpg"))
    mid_min = img.shape[1] // 2 - 100
    mid_max = img.shape[1] // 2 + 100
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = new_img[:, mid_min:mid_max + 1]
    if title == size_title_exception:
        column_sum = image_shape_exception[1] * 255 - np.sum(new_img, axis=0)
    else:
        column_sum = image_shape[1] * 255 - np.sum(new_img, axis=0)
    zeroes = [i[0] for i in np.argwhere(column_sum==0)]
    if len(zeroes) == 0:
        if not (np.min(column_sum) * 402 < np.sum(column_sum)):
            mid = 100
        else:
            mid = np.argmin(column_sum)
        return mid + mid_min
    return zeroes[len(zeroes)//2] + mid_min


def page_annotations_visual(title: str, page_no: int) -> Optional[np.ndarray]:
    data = annotations(title, page_no)
    img = cv2.imread(op.join(root, "manga109", "images", title, f"{page_no:03d}.jpg"))

    if "text" not in data and "frame" not in data:
        return
    frames = data["frame"]
    text_boxes = data["text"]
    if not isinstance(text_boxes, list):
        text_boxes = [text_boxes]

    for box in text_boxes:
        x1 = int(box["@xmin"])
        y1 = int(box["@ymin"])
        x2 = int(box["@xmax"])
        y2 = int(box["@ymax"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not isinstance(frames, list):
        frames = [frames]
    for box in frames:
        x1 = int(box["@xmin"])
        y1 = int(box["@ymin"])
        x2 = int(box["@xmax"])
        y2 = int(box["@ymax"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mid = img.shape[1] // 2
    cv2.rectangle(img, (mid - 100, 0), (mid + 100, img.shape[0]), (0, 0, 225))
    return img


def mask_page(title: str, page_no: int) -> Optional[np.ndarray]:
    data = annotations(title, page_no)
    img = cv2.imread(op.join(root, "manga109", "images", title, f"{page_no:03d}.jpg"))
    mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    if "text" not in data:
        return
    text_boxes = data["text"]
    if isinstance(text_boxes, list):
        for box in text_boxes:
            x1 = box["@xmin"]
            y1 = box["@ymin"]
            x2 = box["@xmax"]
            y2 = box["@ymax"]
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
            cv2.fillPoly(mask, [points], (0, 0, 0))
    else:
        x1 = text_boxes["@xmin"]
        y1 = text_boxes["@ymin"]
        x2 = text_boxes["@xmax"]
        y2 = text_boxes["@ymax"]
        points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
        cv2.fillPoly(mask, [points], (0, 0, 0))
    fg = cv2.bitwise_or(img, img, mask=mask)

    background = np.full(img.shape, 255, dtype=np.uint8)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    return final
