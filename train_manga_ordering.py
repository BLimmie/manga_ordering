import torch
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from data.dataloaders import MangaLoader
from models.aon import AONWithImage
from models.loss.loss_functions import listMLE


def main(epochs=1, batch_size=400, device="cuda:0", lr=5e-3, fine_tuning_lr=5e-5, start_epoch=0, start_index=0,
         max_batch_length=20, with_images=False, patience=3):
    model = AONWithImage(device)
    model.load_state_dict(torch.load("aon_pretrained_epoch_02.pth", map_location=device))

    train_loader = MangaLoader("manga109/sentence_order.json", "manga109/masked_images", images=with_images,
                               split="train")
    val_loader = MangaLoader("manga109/sentence_order.json", "manga109/masked_images", images=with_images,
                             split="validation", shuffle=False)
    optim = AdamW([{'params': model.sentence_encoder.parameters(), 'lr': fine_tuning_lr},
                   {'params': model.page_encoder.parameters()},
                   {'params': model.page_decoder.parameters()}], lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=23,
                                                num_training_steps=epochs * (-(len(train_loader) // -batch_size)))
    frozen = False
    model_type = "aon_naive" if not with_images else "aon_mm"
    min_val_los, best_epoch = -1, -1
    strikes = patience
    for epoch in trange(start_epoch, epochs):
        model.train()
        optim.zero_grad()
        batch_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, y, img) in pbar:
            if i < start_index or x is None:
                continue
            if len(x) > max_batch_length and not frozen:
                for param in model.sentence_encoder.parameters():
                    param.requires_grad = False
            elif len(x) <= max_batch_length and frozen:
                for param in model.sentence_encoder.parameters():
                    param.requires_grad = True
            model_output = model(x, img, pretraining=not with_images)
            y = y.to(device)
            loss = listMLE(model_output[:len(y)], y)
            loss.backward()
            batch_loss += loss.item()
            # Deallocate memory
            del loss
            del y
            if (i + 1) % batch_size == 0 or i + 1 == len(train_loader):
                optim.step()
                scheduler.step()
                optim.zero_grad()
                pbar.set_postfix({"batch_loss": batch_loss, "lr": scheduler.get_last_lr()})
                batch_loss = 0
        with torch.no_grad():
            val_loss = 0
            model.eval()
            for i, (x, y, img) in tqdm(enumerate(val_loader), total=len(val_loader)):
                if x is None:
                    continue
                model_output = model(x, img, pretraining=not with_images)
                y = y.to(device)
                loss = listMLE(model_output, y)
                val_loss += loss.item()
                del loss
                del y
            print(f"Validation Loss - Epoch {epoch}: {val_loss}")
            if val_loss > min_val_los != -1:
                if strikes == 0:
                    break
                strikes -= 1
            else:
                min_val_los = val_loss
                best_epoch = epoch
                strikes = patience
        torch.save(model.state_dict(), f"ckpt/{model_type}_{epoch:02d}.pth")
    print(best_epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--fine_tuning_lr", default=1e-5)
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument("--start_index", default=0)
    parser.add_argument("--max_batch_length", default=60)
    parser.add_argument("--with_images", action="store_true")
    parser.add_argument("--patience", default=3)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        fine_tuning_lr=args.fine_tuning_lr,
        start_epoch=args.start_epoch,
        start_index=args.start_index,
        max_batch_length=args.max_batch_length,
        with_images=args.with_images,
        patience=args.patience
    )
