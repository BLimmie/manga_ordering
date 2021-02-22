from itertools import chain

import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from data.dataloaders import WikiLoader
from models.aon import AONNaive
from models.loss.loss_functions import listMLE


def main(epochs=1, batch_size=400, device="cuda:0", lr=5e-3, fine_tuning_lr=5e-5, start_epoch=0, start_index=0):
    model = AONNaive(device)
    if start_index != 0:
        model.load_state_dict(torch.load(f"ckpt/aon_pretrained_{start_index:08d}.pth"))
    optim = Adam(chain(model.page_encoder.parameters(), (model.page_decoder.parameters())), lr)
    optim_bert = Adam(model.sentence_encoder.parameters(), fine_tuning_lr)
    dataloader = WikiLoader("jawiki/japanese_wiki_paragraphs.json")
    for epoch in trange(start_epoch, epochs):
        optim.zero_grad()
        optim_bert.zero_grad()
        batch_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, y) in pbar:
            print(len(x))
            if i < start_index:
                continue
            model_output = model(x)
            y = y.to(device)
            loss = listMLE(model_output, y)
            loss.backward()
            batch_loss += loss.item()
            # Deallocate memory
            del loss
            del y
            if (i + 1) % batch_size == 0:
                pbar.set_postfix({"batch_loss": batch_loss})
                batch_loss = 0
                optim.step()
                optim_bert.step()
                optim.zero_grad()
                optim_bert.zero_grad()
                torch.cuda.empty_cache()
                if (i+1)//batch_size % 50 == 0:
                    torch.save(model.state_dict(), f"ckpt/aon_pretrained_{i+1:08d}.pth")
        torch.save(model.state_dict(), f"aon_pretrained_epoch_{epoch:02d}.pth")

if __name__ == '__main__':
    main()
