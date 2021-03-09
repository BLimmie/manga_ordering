from transformers import Trainer, TrainingArguments

from data.dataloaders import PairwiseWikiData
from models.tpsortmodels import base_order_model

train_set = PairwiseWikiData("jawiki/japanese_wiki_smoke.json", divisions=10)
model = base_order_model()

training_args = TrainingArguments(
    output_dir="ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=50,
    learning_rate=5e-6,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=len(train_set) // 100 + 1,
    save_steps=len(train_set) // 100 + 1,
    save_total_limit=2,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set
)

trainer.train()