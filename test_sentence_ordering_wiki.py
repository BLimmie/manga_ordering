import torch
from tqdm import tqdm
import csv
from data.dataloaders import WikiLoader
from models.aon import AONNaive
from models.metrics import calculate_metrics


def main(model_path: str, device="cuda:0"):
    model = AONNaive(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    dataloader = WikiLoader("jawiki/japanese_wiki_paragraphs.json", divisions=50, offset=37)
    kt = 0
    pmr = 0
    results = [["Num Sentences", "KT", "PMR"]]
    nones = 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if x is None:
                nones += 1
                continue

            model_output = model(x)[:len(y)]
            y = y.to(device)
            metrics = calculate_metrics(model_output, y)
            results.append([len(x), metrics["Kendall's Tau"], metrics["Perfect Match Ratio"]])
            pmr += metrics["Perfect Match Ratio"]
            kt += metrics["Kendall's Tau"]

    print(f"Total Kendall's Tau: {kt}")
    print(f"Total PMR: {pmr}")
    print(f"Average Kendall's Tau: {kt / (len(dataloader) - nones)}")
    print(f"Average PMR: {pmr / (len(dataloader) - nones)}")

    with open('WikiResults.csv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(results)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    main(args.model_path, args.device)
