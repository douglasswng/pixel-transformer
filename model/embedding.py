import math
import torch
import torch.nn as nn
from constants import TOKEN_COUNT, MODEL_DIM, PAD_ID, MAX_SEQ_LEN

class Embedding(nn.Module):
    def __init__(self, token_count: int = TOKEN_COUNT, model_dim: int = MODEL_DIM):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, model_dim, padding_idx=PAD_ID)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        self.model_dim = model_dim

    def forward(self, input_ids):
        return self.token_embedding(input_ids) * math.sqrt(self.model_dim)
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, model_dim: int = MODEL_DIM, max_seq_length: int = MAX_SEQ_LEN):
        super().__init__()
        self.model_dim = model_dim
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_seq_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

    
if __name__ == '__main__':
    embedding = Embedding()
    from dataloader.loader import val_loader
    for batch in val_loader:
        encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
        print(embedding(encoder_input_ids).shape)
        print(embedding(decoder_input_ids).shape)
        break