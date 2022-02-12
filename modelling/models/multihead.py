from turtle import forward
import torch
from torch import nn

class MultiHeadModel(nn.Module):
    def __init__(self, encoder, decoder_heads):
        super().__init__()
        self.encoder = encoder
        if type(decoder_heads) is list:
            self._decoder_type = list
            self.decoders = nn.ModuleList(decoder_heads)
        elif type(decoder_heads) is dict:
            self._decoder_type = dict
            self.decoders = nn.ModuleDict(decoder_heads)
        else:
            raise RuntimeError("Unrecognized decoder head type!")

    def forward(self, inputs):
        latent = self.encoder(inputs)
        if self._decoder_type is list:
            return [decoder(latent) for decoder in self.decoders]
        elif self._decoder_type is dict:
            return {key: self.decoders[key](latent) for key in self.decoders}

