import torch
from tqdm import tqdm
import csv

from transformers import TrainingArguments, Trainer

from data.dataloaders import WikiLoader, SentenceDataset
from models.aon import AONNaive
from models.topological_sort import convert_to_graph
from models.tpsortmodels import base_order_model
from models.metrics import calculate_metrics


def main(model_path: str, device="cuda:0", pairwise=False):
    if pairwise:
        model = base_order_model()
        training_args = TrainingArguments(
            output_dir='./dummy_folder',
            per_device_eval_batch_size=120,
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
    else:
        model = AONNaive(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    dataloader = WikiLoader("jawiki/japanese_wiki_paragraphs.json", divisions=500, offset=37, test=True)
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
            if pairwise:
                ds = SentenceDataset(x)
                res = tester.predict(ds)
                res = res.predictions
                if len(res.shape) > 1:
                    res = res.argmax(axis=1) * 2 - 1
                metrics = convert_to_graph(res, ds.pos)
            else:
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
    parser.add_argument("--pairwise", action="store_true")
    args = parser.parse_args()
    main(args.model_path, args.device, args.pairwise)
