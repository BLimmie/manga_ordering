import json

import torch
from tqdm import tqdm
import csv

from transformers import TrainingArguments, Trainer

from data.dataloaders import WikiLoader, SentenceDataset, PageDataset
from data.pages import get_pages_partition
from models.aon import AONNaive
from models.topological_sort import convert_to_graph
from models.tpsortmodels import MMOrderModel
from models.metrics import calculate_metrics


def main(model_path: str, device="cuda:0"):

    model = MMOrderModel()
    training_args = TrainingArguments(
        output_dir='./dummy_folder',
        per_device_eval_batch_size=25,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy="epoch",
        disable_tqdm=True
    )
    tester = Trainer(model=model, args=training_args)
    with open("manga109/sentence_order.json") as f:
        manga_data = json.load(f)
    model.load_state_dict(torch.load(model_path, map_location=device))
    pages = get_pages_partition(with_text=True, split="test")
    kt = 0
    pmr = 0
    results = [["Num Sentences", "KT", "PMR"]]
    nones = 0
    model.eval()
    with torch.no_grad():
        for i, (page) in tqdm(enumerate(pages), total=len(pages)):
            try:
                ds = PageDataset(manga_data, "manga109/masked_images", page)
                res = tester.predict(ds)
                res = res.predictions
                if len(res.shape) > 1:
                    res = (res.argmax(axis=1) * 2 - 1)
                metrics = convert_to_graph(res, ds.pos)
                results.append([len(ds), metrics["Kendall's Tau"], metrics["Perfect Match Ratio"]])
                pmr += metrics["Perfect Match Ratio"]
                kt += metrics["Kendall's Tau"]
            except Exception as e:
                print(e)
                nones += 1


    print(f"Total Kendall's Tau: {kt}")
    print(f"Total PMR: {pmr}")
    print(f"Average Kendall's Tau: {kt / (len(pages) - nones)}")
    print(f"Average PMR: {pmr / (len(pages) - nones)}")

    with open('masked.csv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(results)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.model_path, args.device)
