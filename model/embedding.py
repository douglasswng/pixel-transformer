import torch.nn as nn
from constants import TOKEN_COUNT, MODEL_DIM

class Embedding(nn.Module):
    def __init__(self, token_count: int = TOKEN_COUNT, model_dim: int = MODEL_DIM):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, model_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

    def forward(self, input_ids):
        return self.token_embedding(input_ids)
    
if __name__ == '__main__':
    embedding = Embedding()
    from dataloader.loader import val_loader
    for batch in val_loader:
        encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
        print(embedding(encoder_input_ids).shape)
        print(embedding(decoder_input_ids).shape)
        break