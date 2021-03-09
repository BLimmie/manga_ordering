import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel
from torchvision.models import resnet18
from .encoders import FFNN
from torch.nn import BCEWithLogitsLoss
model_base = "cl-tohoku/bert-base-japanese"

def base_order_model() -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(model_base, num_labels=2)

class MMOrderModel(nn.Module):
    def __init__(self, image_encoding_size=768, load_from_pretrained=None,device="cuda:0"):
        super(MMOrderModel, self).__init__()
        self.base_model = base_order_model()
        if load_from_pretrained is not None:
            self.base_model.load_state_dict(torch.load(load_from_pretrained))
        self.base_model = self.base_model.base_model
        self.image_model = resnet18(num_classes=image_encoding_size)
        self.classifier = FFNN(input_dim=768+image_encoding_size, num_layers=8, score=False)
        self.loss_fn = BCEWithLogitsLoss()
        self.base_model = self.base_model.to(device)
        self.image_model = self.image_model.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
    def forward(self, image, *args, **kwargs):
        for key, t in kwargs.items():
            kwargs[key] = t.to(self.device)
        image = image.to(self.device)
        labels = kwargs.pop("labels", None)
        image_encoding = self.image_model(image)
        sentences_encoding = self.base_model(*args, **kwargs).pooler_output
        combined = torch.cat([image_encoding, sentences_encoding],dim=1)
        output = self.classifier(combined).squeeze(1)
        if labels is not None:
            labels = labels.type(torch.float32)
            loss = self.loss_fn(output, labels)
            return loss, output
        else:
            return output
