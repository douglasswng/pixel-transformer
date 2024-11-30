import torch.nn as nn
from model.embedding import Embedding
from model.attention import MultiHeadAttention
from model.ffn import FFNLayer
from constants import MODEL_DIM, LAYERS

class DecoderBlock(nn.Module):
    def __init__(self, model_dim: int = MODEL_DIM):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.cross_attention = MultiHeadAttention()
        self.ffn = FFNLayer()
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output, pad=None, encoder_pad=None):
        residual = x.clone()
        x = self.self_attention(x, pad=pad, causal_mask=True)
        x = x + residual
        x = self.norm1(x)

        residual = x.clone()
        x = self.cross_attention(x, encoder_output, pad=pad, context_pad=encoder_pad)
        x = x + residual
        x = self.norm2(x)

        residual = x.clone()
        x = self.ffn(x)
        x = x + residual
        x = self.norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, layers: int = LAYERS):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([
            DecoderBlock() 
            for _ in range(layers)
        ])

    def forward(self, x, encoder_output, pad=None, encoder_pad=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, pad=pad, encoder_pad=encoder_pad)
        return x