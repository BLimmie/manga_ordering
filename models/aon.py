import torch.nn as nn
import torch

from .encoders import sentence_encoder_model, naive_page_encoder_decoder
from .preprocessing.tokenization import tokenize
class AONNaive(nn.Module):
    def __init__(self):
        super(AONNaive, self).__init__()
        self.sentence_encoder = sentence_encoder_model()
        self.page_encoder, self.page_decoder = naive_page_encoder_decoder()

    def forward(self, page_texts):
        input_tokens = tokenize(page_texts)
        encoded_sentences = self.sentence_encoder(**input_tokens).pooler_output.unsqueeze(0)

        page_encoding = self.page_encoder(inputs_embeds = encoded_sentences, attention_mask = torch.ones((1,len(encoded_sentences)))).last_hidden_state.squeeze(0)
        return self.page_decoder(page_encoding)