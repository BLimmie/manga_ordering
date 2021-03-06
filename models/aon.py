from typing import Union

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from .encoders import sentence_encoder_model, naive_page_encoder_decoder
from .preprocessing.tokenization import tokenize


class AONWithImage(nn.Module):
    def __init__(self, device: Union[torch.device, str] = "cpu", image_vectors: int = 8):
        super(AONWithImage, self).__init__()
        self.image_vectors = image_vectors
        self.image_encoder_output_size = 768 * image_vectors
        self.sentence_encoder = sentence_encoder_model()
        self.page_encoder, self.page_decoder = naive_page_encoder_decoder()
        self.image_encoder = resnet18(num_classes=self.image_encoder_output_size)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.to(self.device)
        self.sentence_encoder = self.sentence_encoder.to(self.device)
        self.page_encoder = self.page_encoder.to(self.device)
        self.page_decoder = self.page_decoder.to(self.device)

    def forward(self, page_texts, image, pretraining=False):
        input_tokens = tokenize(page_texts)
        for key, tensor in input_tokens.items():
            if tensor.shape[1] > 512:
                tensor = tensor[:, :512]
            input_tokens[key] = tensor.to(self.device)
        minibatches = []
        if len(input_tokens["input_ids"] > 5):
            input_ids_split = torch.split(input_tokens["input_ids"], 5)
            token_ids_split = torch.split(input_tokens["token_type_ids"], 5)
            attention_mask_split = torch.split(input_tokens["attention_mask"], 5)
            num_splits = len(input_tokens["input_ids"]) // 5 + int(len(input_tokens["input_ids"]) % 5 != 0)
            for i in range(num_splits):
                minibatches.append({"input_ids": input_ids_split[i],
                                    "token_type_ids": token_ids_split[i],
                                    "attention_mask": attention_mask_split[i]})
        else:
            minibatches.append(input_tokens)
        encoded_sentences = []
        for minibatch in minibatches:
            encoded_sentences.append(self.sentence_encoder(**minibatch).pooler_output.unsqueeze(0))
        encoded_sentences = torch.cat(encoded_sentences, dim=1)
        if pretraining:
            attention_mask = torch.ones((1, encoded_sentences.shape[1]), device=self.device)
        else:
            attention_mask = torch.ones((1, encoded_sentences.shape[1] + self.image_vectors),
                                        device=self.device)

        if pretraining:
            token_to_id = None
        else:
            token_to_id = torch.cat((torch.zeros((1, encoded_sentences.shape[1]), device=self.device, dtype=torch.long),
                                     torch.ones((1, self.image_vectors), device=self.device, dtype=torch.long)), dim=1)

        if not pretraining:
            assert image is not None
            image = image.to(self.device)
            image_vectors = self.image_encoder(image).view(1, -1, 768)
            encoded_sentences = torch.cat([encoded_sentences, image_vectors], dim=1)
        page_encoding = self.page_encoder(inputs_embeds=encoded_sentences,
                                          token_type_ids=token_to_id,
                                          attention_mask=attention_mask).last_hidden_state.squeeze(0)

        return self.page_decoder(page_encoding)


class AONNaive(AONWithImage):
    def forward(self, page_texts):
        return super().forward(page_texts, None, pretraining=True)
