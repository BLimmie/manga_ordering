from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from data.dataloaders import PairwiseMangaDataNoIMG, PairwiseMangaData
from models.tpsortmodels import base_order_model, MMOrderModel

train_set = PairwiseMangaData("manga109/sentence_order.json", "manga109/images", split="train")
valid_set = PairwiseMangaData("manga109/sentence_order.json", "manga109/images", split="validation")
model = MMOrderModel(load_from_pretrained="pytorch_model.bin")

training_args = TrainingArguments(
    output_dir="ckpt",
    num_train_epochs=2,
    per_device_train_batch_size=5,
    learning_rate=5e-5,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=len(train_set) // 25 + 1,
    save_steps=len(train_set) // 25 + 1,
    save_total_limit=2,
    dataloader_num_workers=6
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set
)
torch.save(model.state_dict(), "unmasked.pth")
trainer.train()