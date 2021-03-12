from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from data.dataloaders import PairwiseMangaDataNoIMG
from models.tpsortmodels import base_order_model, MMOrderModel

train_set = PairwiseMangaDataNoIMG("manga109/sentence_order.json", split="train")
valid_set = PairwiseMangaDataNoIMG("manga109/sentence_order.json", split="validation")
model = base_order_model()

training_args = TrainingArguments(
    output_dir="ckpt",
    num_train_epochs=100,
    per_device_train_batch_size=40,
    learning_rate=5e-6,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=len(train_set) // 200 + 1,
    save_steps=len(train_set) // 200 + 1,
    save_total_limit=2,
    dataloader_num_workers=8,
    evaluation_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    callbacks=[EarlyStoppingCallback(3)]
)
torch.save(model.state_dict(), "naive.pth")
trainer.train()