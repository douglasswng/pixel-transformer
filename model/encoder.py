import torch.nn as nn
from model.embedding import Embedding
from model.attention import MultiHeadAttention
from model.ffn import FFNLayer
from constants import MODEL_DIM, LAYERS

class EncoderBlock(nn.Module):
    def __init__(self, model_dim: int = MODEL_DIM):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.ffn = FFNLayer()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, pad=None):
        residual = x.clone()
        x = self.attention(x, pad=pad)
        x = x + residual
        x = self.norm1(x)

        residual = x.clone()
        x = self.ffn(x)
        x = x + residual
        x = self.norm2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: int = LAYERS):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(layers)])

    def forward(self, x, pad=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, pad=pad)
        return x