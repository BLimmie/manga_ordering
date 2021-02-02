from tqdm import tqdm
import json
from data.image_processing import find_midpoint, page_annotations_visual, frame_crosses_middle
from data.data_processing import text_in_frames, frame_order
import cv2
from data.pages import get_pages

pages = get_pages(with_text=True)
test = 5877
test_page = pages[test] #19
idx = 6500
# print(json.dumps(text_in_frames(*test_page), indent=2, ensure_ascii=False))
pages = pages[idx:]
annotations = page_annotations_visual(*test_page)
#00
# if test-idx > 500:
#     mid = find_midpoint(*test_page)
#     cv2.line(annotations, (mid, 0), (mid, annotations.shape[0]), (255,0,255))
#     cv2.imshow("",annotations)
#     cv2.waitKey(0)

# print(json.dumps(text_in_frames(*test_page), indent=2, ensure_ascii=False))

frame_order(*test_page)
for i, page in tqdm(enumerate(pages), total=len(pages)):
    i=i+idx
    frame_order(*page)