import operator
from typing import Any, List, Dict

from data.annotations import annotations
from data.constants import yonkoma
from data.image_processing import frame_crosses_middle, find_midpoint


def text_order(title, page_no) -> List[str]:
    frames = frame_order(title, page_no)
    text = text_in_frames(title,page_no)
    final_order = []
    for frame in frames:
        final_order.extend(text_order_in_frame(text[frame]))
    return final_order


def text_order_in_frame(text_boxes: List[Dict[str,Any]]) -> List[str]:

    if len(text_boxes) == 0:
        return []
    id_to_text = {}
    for text in text_boxes:
        id = text["@id"]
        id_to_text[id] = text["#text"]
    frames_dict = {}
    x_max = -1
    x_min = 2000
    y_max = -1
    y_min = 2000
    # Find the min
    for frame in text_boxes:
        if int(frame["@xmin"]) < x_min:
            x_min = int(frame["@xmin"])
        if int(frame["@ymin"]) < y_min:
            y_min = int(frame["@ymin"])
        if int(frame["@xmax"]) > x_max:
            x_max = int(frame["@xmax"])
        if int(frame["@ymax"]) > y_max:
            y_max = int(frame["@ymax"])

    for frame in text_boxes:
        id = frame["@id"]
        topright = (int(frame["@xmax"]), int(frame["@ymin"]))
        botleft = (int(frame["@xmin"]), int(frame["@ymax"]))
        n_tr = (round((topright[0] - x_min) ), round((topright[1] - y_min)))
        n_bl = (round((botleft[0] - x_min) ), round((botleft[1] - y_min)))
        next_center = ((n_tr[0]+n_bl[0])/2, (n_tr[1]+n_bl[0])/2)
        frames_dict[id] = (n_tr, n_bl, next_center)

    toprightmost = list(frames_dict.keys())[0]
    toprightmostvalue = ((y_max-y_min) - frames_dict[toprightmost][0][1]) + frames_dict[toprightmost][0][0]
    for id, (tr, bl, c) in frames_dict.items():
        value = ((y_max-y_min) - tr[1]) + tr[0]
        if value == toprightmostvalue and tr[1] < frames_dict[toprightmost][0][1]:
            toprightmost = id
        elif value > toprightmostvalue:
            toprightmost = id
            toprightmostvalue = value

    visited = set()
    text_box_order = []
    visited.add(toprightmost)
    text_box_order.append(toprightmost)
    cur_center = frames_dict[toprightmost][2]
    while len(visited) < len(text_boxes):
        min_dist = 99999999
        next_box = None
        next_center = None
        for id, (tr, bl, c) in frames_dict.items():
            if id in visited:
                continue
            dist = ((cur_center[0] - c[0])**2 + (cur_center[1] - c[1])**2)**0.5
            if dist < min_dist:
                next_box = id
                min_dist = dist
                next_center = c
        visited.add(next_box)
        text_box_order.append(next_box)
        cur_center = next_center

    final_order = []
    for box_id in text_box_order:
        final_order.append(id_to_text[box_id])
    return final_order

def text_in_frames(title, page_no) -> Dict[str, List[Dict[str, Any]]]:
    data = annotations(title, page_no)
    texts = data["text"]
    if not isinstance(texts, list):
        texts = [texts]
    if "frame" not in data:
        return {"-1": [text for text in texts]}
    frames = data["frame"]
    if not isinstance(frames, list):
        frames = [frames]
    for i in range(len(frames)):
        frames[i]["boxtopleft"] = (int(frames[i]["@xmin"]), int(frames[i]["@ymin"]))
        frames[i]["boxbotright"] = (int(frames[i]["@xmax"]), int(frames[i]["@ymax"]))
    frames_to_text = {frame["@id"]: [] for frame in frames}
    for text in texts:
        text_topleft = (int(text["@xmin"]), int(text["@ymin"]))
        text_botright = (int(text["@xmax"]), int(text["@ymax"]))
        areas = {frame["@id"]: 0 for frame in frames}
        for frame in frames:
            frame_topleft = frame["boxtopleft"]
            frame_botright = frame["boxbotright"]
            SI = max(0, min(text_botright[0], frame_botright[0]) - max(text_topleft[0], frame_topleft[0])) * \
                 max(0, min(text_botright[1], frame_botright[1]) - max(text_topleft[1], frame_topleft[1]))
            areas[frame["@id"]] = SI
        best_frame = max(areas.items(), key=operator.itemgetter(1))[0]
        if areas[best_frame] == 0:
            continue
        frames_to_text[best_frame].append(text)
    return frames_to_text


def frame_order(title, page_no) -> List[str]:
    data = annotations(title, page_no)
    text = text_in_frames(title, page_no)
    if "frame" not in data:
        return ["-1"]
    frames = data["frame"]
    if not isinstance(frames, list):
        frames = [frames]
    if title in yonkoma:
        return _frame_order_yonkoma(frames)
    frames = [f for f in frames if len(text[f["@id"]]) > 0]

    if len(frames) == 0:
        return ["-1"]

    if frame_crosses_middle(title, page_no):
        return _frame_order(frames)
    midpoint = find_midpoint(title, page_no)
    right_page_frames = []
    left_page_frames = []
    for frame in frames:
        topright_x = int(frame["@xmax"])
        # Have a small buffer to fix pixel errors
        if topright_x > midpoint + 50:
            right_page_frames.append(frame)
        else:
            left_page_frames.append(frame)
    return _frame_order(right_page_frames) + _frame_order(left_page_frames)


def _frame_order_yonkoma(frames):
    mesh = 8
    if len(frames) == 0:
        return []
    frames_dict = {}
    x_max = -1
    x_min = 2000
    y_max = -1
    y_min = 2000
    # Find the min
    for frame in frames:
        if int(frame["@xmin"]) < x_min:
            x_min = int(frame["@xmin"])
        if int(frame["@ymin"]) < y_min:
            y_min = int(frame["@ymin"])
        if int(frame["@xmax"]) > x_max:
            x_max = int(frame["@xmax"])
        if int(frame["@ymax"]) > y_max:
            y_max = int(frame["@ymax"])
    # d is the distance between lines on the mesh
    d_x = (x_max - x_min) / mesh
    d_y = (y_max - y_min) / mesh
    for frame in frames:
        id = frame["@id"]
        topright = (int(frame["@xmax"]), int(frame["@ymin"]))
        botleft = (int(frame["@xmin"]), int(frame["@ymax"]))
        n_tr = (round((topright[0] - x_min) / d_x), round((topright[1] - y_min) / d_y))
        n_bl = (round((botleft[0] - x_min) / d_x), round((botleft[1] - y_min) / d_y))
        frames_dict[id] = (n_tr, n_bl)

    a = sorted(frames_dict.items(), key=lambda x: x[1][0][0] * mesh - x[1][0][1], reverse=True)
    return [id for (id, _) in a]


def _frame_order(frames,mesh=20) -> List[str]:
    # id : ((top right point),(bottom left point))
    if len(frames) == 0:
        return []
    frames_dict = {}
    x_max = -1
    x_min = 2000
    y_max = -1
    y_min = 2000
    # Find the min
    for frame in frames:
        if int(frame["@xmin"]) < x_min:
            x_min = int(frame["@xmin"])
        if int(frame["@ymin"]) < y_min:
            y_min = int(frame["@ymin"])
        if int(frame["@xmax"]) > x_max:
            x_max = int(frame["@xmax"])
        if int(frame["@ymax"]) > y_max:
            y_max = int(frame["@ymax"])
    # d is the distance between lines on the mesh
    d_x = (x_max - x_min) / mesh
    d_y = (y_max - y_min) / mesh
    for frame in frames:
        id = frame["@id"]
        topright = (int(frame["@xmax"]), int(frame["@ymin"]))
        botleft = (int(frame["@xmin"]), int(frame["@ymax"]))
        n_tr = (round((topright[0] - x_min) / d_x), round((topright[1] - y_min) / d_y))
        n_bl = (round((botleft[0] - x_min) / d_x), round((botleft[1] - y_min) / d_y))
        frames_dict[id] = (n_tr, n_bl)

    frames_after = {}
    for frame_id, (tr, bl) in frames_dict.items():
        candidate_afters = []
        for frame_id2, (tr2, bl2) in frames_dict.items():

            if frame_id == frame_id2:
                continue
            # Frame 1 is above Frame 2
            if tr2[1] >= bl[1]:
                for frame_id3, (tr3, bl3) in frames_dict.items():
                    if frame_id3 == frame_id2 or frame_id3 == frame_id:
                        continue
                    if bl3[0] > bl[0] and tr3[0] < tr2[0] and tr3[1] < (bl[1] + tr[1]) / 2 and bl3[1] > (
                            tr2[1] + bl2[1]) / 2:
                        break
                else:
                    candidate_afters.append(frame_id2)
                    continue
            # opposite
            if tr[1] >= bl2[1]:
                for frame_id3, (tr3, bl3) in frames_dict.items():
                    if frame_id3 == frame_id2 or frame_id3 == frame_id:
                        continue
                    if bl3[0] > bl2[0] and tr3[0] < tr[0] and tr3[1] < (bl2[1] + tr2[1]) / 2 and bl3[1] > (
                            tr[1] + bl[1]) / 2:
                        break
                else:
                    continue

            # Frame 2 is to the left of Frame 1 or slightly to the left of Frame 1
            if bl[0] >= tr2[0] or (tr2[0] < tr[0] and bl2[0] < bl[0]):
                # If one of the frame's y values are fully enclosed by the other, then it's a candidate for being
                # before the other
                if bl[1] >= bl2[1] and tr[1] <= tr2[1]:
                    candidate_afters.append(frame_id2)
                    continue
                if bl2[1] >= bl[1] and tr2[1] <= tr[1]:
                    candidate_afters.append(frame_id2)
                    continue

                # Frame 1 is slightly above Frame 2
                if tr[1] < tr2[1] and bl[1] <= bl2[1]:
                    candidate_afters.append(frame_id2)
                    continue

                # Frame 1 is slightly below Frame 2 but not completely
                if tr2[1] < tr[1] < bl2[1] <= bl[1]:
                    for frame_id3, (tr3, bl3) in frames_dict.items():
                        if frame_id3 == frame_id2 or frame_id3 == frame_id:
                            continue
                        if bl3[0] > bl2[0] and tr3[0] < tr[0] and tr3[1] < (bl2[1] + tr2[1]) / 2 and bl3[1] > (
                                tr[1] + bl[1]) / 2:
                            break
                    else:
                        # If there is no frame in between
                        if tr[1] <= (tr2[1] + bl2[1]) / 2:
                            candidate_afters.append(frame_id2)
                            continue
                        else:
                            continue
                    # If there is a frame between, Frame 2 comes after Frame 1
                    candidate_afters.append(frame_id2)
                    continue

            # opposite
            if bl2[0] >= tr[0] or (tr[0] < tr2[0] and bl[0] < bl2[0]):
                if bl2[1] >= bl[1] and tr2[1] <= tr[1]:
                    continue
                if bl[1] >= bl2[1] and tr[1] <= tr2[1]:
                    continue

                if tr2[1] < tr[1] and bl2[1] <= bl[1]:
                    continue

                if tr[1] < tr2[1] < bl[1] <= bl2[1]:
                    for frame_id3, (tr3, bl3) in frames_dict.items():
                        if frame_id3 == frame_id2 or frame_id3 == frame_id:
                            continue
                        if bl3[0] > bl[0] and tr3[0] < tr2[0] and tr3[1] < (bl[1] + tr[1]) / 2 and bl3[1] > (
                                tr[1] + bl[1]) / 2:
                            break
                    else:
                        if tr2[1] <= (tr[1] + bl[1]) / 2:
                            continue
                        else:
                            candidate_afters.append(frame_id2)
                            continue
                    continue

            # Frame 2 is overlapping vertically with Frame 1
            if bl2[1] > bl[1] and tr2[1] >= tr[1] and (
                    (tr[0] >= tr2[0] and bl[0] <= bl2[0]) or (tr[0] <= tr2[0] and bl[0] >= bl2[0])):
                candidate_afters.append(frame_id2)
                continue
            # opposite
            if bl[1] > bl2[1] and tr[1] >= tr2[1] and (
                    (tr2[0] >= tr[0] and bl2[0] <= bl[0]) or (tr2[0] <= tr[0] and bl2[0] >= bl[0])):
                continue

            # Cross pattern, Frame 1 is the horizontal cross, Frame 2 is the vertical cross
            if tr[1] > tr2[1] and bl[1] < bl2[1] and tr[0] >= tr2[0] and bl[0] <= bl2[0]:
                if tr[1] <= (tr2[1] + bl2[1]) / 2:
                    candidate_afters.append(frame_id2)
                    continue
            if tr2[1] > tr[1] and bl2[1] < bl[1] and tr2[0] >= tr[0] and bl2[0] <= bl[0]:
                if tr[1] <= (tr2[1] + bl2[1]) / 2:
                    continue

            # Frame 1 is enclosed by Frame 2
            if tr[1] >= tr2[1] and bl[1] <= bl2[1] and tr[0] <= tr2[0] and bl[0] >= bl2[0]:
                candidate_afters.append(frame_id2)
                continue
            if tr2[1] >= tr[1] and bl2[1] <= bl[1] and tr2[0] <= tr[0] and bl2[0] >= bl[0]:
                continue
        frames_after[frame_id] = candidate_afters
    a = set(frames_after.keys())
    b = set().union(*frames_after.values())
    assert len(a) == len(b) + 1
    L = []
    S = a - b
    while len(S) > 0:
        n = S.pop()
        L.append(n)
        del frames_after[n]
        if len(S) == 0:
            a = set(frames_after.keys())
            b = set().union(*frames_after.values())
            S = a - b
    assert len(L) == len(frames)
    return L
