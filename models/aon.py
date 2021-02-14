from typing import Union

import torch
import torch.nn as nn

from .encoders import sentence_encoder_model, naive_page_encoder_decoder
from .preprocessing.tokenization import tokenize


class AONNaive(nn.Module):
    def __init__(self, device: Union[torch.device, str]):
        super(AONNaive, self).__init__()
        self.sentence_encoder = sentence_encoder_model()
        self.page_encoder, self.page_decoder = naive_page_encoder_decoder()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.to(self.device)
        self.sentence_encoder = self.sentence_encoder.to(self.device)
        self.page_encoder = self.page_encoder.to(self.device)
        self.page_decoder = self.page_decoder.to(self.device)

    def forward(self, page_texts):
        input_tokens = tokenize(page_texts)
        for key, tensor in input_tokens.items():
            input_tokens[key] = tensor.to(self.device)
        encoded_sentences = self.sentence_encoder(**input_tokens).pooler_output.unsqueeze(0)

        page_encoding = self.page_encoder(inputs_embeds=encoded_sentences, attention_mask=torch.ones(
            (1, len(encoded_sentences)), device=self.device)).last_hidden_state.squeeze(0)
        return self.page_decoder(page_encoding)
