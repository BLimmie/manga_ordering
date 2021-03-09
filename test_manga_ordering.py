import csv

import torch
from tqdm import tqdm

from data.dataloaders import MangaLoader
from models.aon import AONWithImage
from models.metrics import calculate_metrics


def main(model_path: str, device="cuda:0"):
    if model_path.find("mm") == -1:
        with_images = False
    else:
        with_images = True

    masked_images = not model_path.find("unmasked") != -1
    image_dir = "masked_images" if masked_images else "images"
    model = AONWithImage(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = MangaLoader("manga109/sentence_order.json", f"manga109/{image_dir}", images=with_images,
                              split="test", shuffle=False)
    kt = 0
    pmr = 0
    nones = 0
    results = [["Num Sentences", "KT", "PMR"]]
    model = model.eval()
    with torch.no_grad():
        for i, (x, y, img) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if x is None:
                nones += 1
                continue
            model_output = model(x, img, pretraining=not with_images)[:len(y)]
            y = y.to(device)
            metrics = calculate_metrics(model_output, y)
            pmr += metrics["Perfect Match Ratio"]
            kt += metrics["Kendall's Tau"]
            results.append([len(x), metrics["Kendall's Tau"], metrics["Perfect Match Ratio"]])


    print(f"Total Kendall's Tau: {kt}")
    print(f"Total PMR: {pmr}")
    print(f"Average Kendall's Tau: {kt / (len(test_loader) - nones)}")
    print(f"Average PMR: {pmr / (len(test_loader) - nones)}")
    with open(f'MangaResults_{"image" if with_images else "naive"}{"masked" if masked_images else ""}.csv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(results)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    main(args.model_path, args.device)
