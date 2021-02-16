import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from data.dataloaders import WikiLoader
from models.aon import AONNaive
from models.loss.loss_functions import listwise_ranking_loss
from itertools import chain
def main(epochs=1, batch_size=32, device="cuda:0", lr=0.01, fine_tuning_lr=1e-6):
    model = AONNaive(device)
    dataloader = WikiLoader("jawiki/japanese_wiki_paragraphs.json")
    optim = Adam(chain(model.page_encoder.parameters(), (model.page_decoder.parameters())), lr)
    optim_bert = Adam(model.sentence_encoder.parameters(), fine_tuning_lr)
    for epoch in trange(epochs):
        optim.zero_grad()
        optim_bert.zero_grad()
        batch_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, y) in pbar:
            model_output = model(x)
            y = y.to(device)
            loss = listwise_ranking_loss(model_output, y)
            loss.backward()
            batch_loss += loss.item()
            #Deallocate memory
            del loss
            del y
            if (i + 1) % batch_size == 0:
                pbar.set_postfix({"batch_loss": batch_loss})
                batch_loss = 0
                optim.step()
                optim_bert.step()
                optim.zero_grad()
                optim_bert.zero_grad()
    torch.save(model.state_dict(), "aon_naive.pth")
if __name__ == '__main__':
    main()