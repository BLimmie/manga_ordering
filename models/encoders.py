from typing import Tuple

from transformers import BertModel, EncoderDecoderConfig, BertConfig
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from torch.nn import Linear
import torch.nn as nn
class FFNN(nn.Module):
    def __init__(self, input_dim = 768, num_layers = 4):
        assert num_layers > 0, "FFNN cannot have non-positive layers"
        super(FFNN, self).__init__()
        self.layers = [Linear(input_dim,input_dim) for _ in range(num_layers-1)]
        self.fc = Linear(input_dim, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        output = self.gelu(self.fc(x))
        return output


def sentence_encoder_model() -> BertModel:
    return BertModel.from_pretrained("cl-tohoku/bert-base-japanese")


def naive_page_encoder_decoder() -> Tuple[BertModel, FFNN]:
    encode_conf = BertConfig(num_hidden_layers=4)
    return BertModel(encode_conf), FFNN(num_layers=4)