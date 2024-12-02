import math
import torch
import torch.nn as nn
from constants import TOKEN_COUNT, MODEL_DIM, PAD_ID, MAX_SEQ_LEN, SPECIAL_TOKEN_COUNT, IMG_H, IMG_W, SPECIAL_TOKEN_IDS

class Embedding(nn.Module):
    def __init__(self, token_count: int = TOKEN_COUNT, model_dim: int = MODEL_DIM):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, model_dim, padding_idx=PAD_ID)
        self.model_dim = model_dim
        self.pos_embedding = SinusoidalPositionalEmbedding()
        self.pos_embedding2d = self._get_pos_embedding2d()
        self.pos_linear = nn.Linear(model_dim, model_dim)
        self._init_weight()

    def _init_weight(self):
        nn.init.trunc_normal_(self.token_embedding.weight, mean=0, std=0.03)
        nn.init.trunc_normal_(self.pos_linear.weight, mean=0, std=0.03)

        with torch.no_grad():
            for token_id in range(self.token_embedding.weight.shape[0]):
                if token_id < SPECIAL_TOKEN_COUNT:
                    continue

                adjusted_id = token_id - SPECIAL_TOKEN_COUNT
                x = adjusted_id % IMG_W
                y = adjusted_id // IMG_W

                self.token_embedding.weight[token_id] = self.pos_embedding2d[:, y, x] / math.sqrt(self.model_dim)

        self.token_embedding.weight.requires_grad = False
        self.token_embedding.weight[:SPECIAL_TOKEN_COUNT].requires_grad = True

    def _get_pos_embedding2d(self):
        pe = torch.zeros(self.model_dim, IMG_H, IMG_W)
        d_model = int(self.model_dim / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., IMG_W).unsqueeze(1)
        pos_h = torch.arange(0., IMG_H).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, IMG_H, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, IMG_H, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, IMG_W)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, IMG_W)

        return pe
    
    def forward(self, input_ids):
        return self.token_embedding(input_ids) * math.sqrt(self.model_dim) + self.pos_linear(self.pos_embedding(input_ids))
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, model_dim: int = MODEL_DIM, max_seq_length: int = MAX_SEQ_LEN):
        super().__init__()
        self.model_dim = model_dim
        
        position = torch.arange(2 * max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(2 * max_seq_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

    
if __name__ == '__main__':
    embedding = Embedding()
    from dataloader.loader import create_dataloaders
    train_loader, val_loader = create_dataloaders()
    for batch in val_loader:
        encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
        print(embedding(encoder_input_ids).shape)
        print(embedding(decoder_input_ids).shape)
        break