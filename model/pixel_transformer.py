import torch
from torch import nn

from model.encoder import Encoder
from model.decoder import Decoder

from constants import MODEL_DIM, TOKEN_COUNT, MAX_SEQ_LEN, DEVICE, START_ID, END_ID, TOKEN_COUNT, IMG_H, IMG_W, SPECIAL_TOKEN_IDS, SPECIAL_TOKEN_COUNT, PAD_ID

class PixelTransformer(nn.Module):
    def __init__(self, model_dim: int = MODEL_DIM, token_count: int = TOKEN_COUNT):
        super().__init__()
        self.token_count = token_count
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.classifier = nn.Linear(model_dim, token_count).to(DEVICE)
        self._init_weight()

    def _init_weight(self):
        self.encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
        self.classifier.weight = self.decoder.embedding.token_embedding.weight

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_pad, decoder_pad, label_ids):
        encoder_output = self.encoder(encoder_input_ids.to(DEVICE), pad=encoder_pad.to(DEVICE))
        decoder_output = self.decoder(decoder_input_ids.to(DEVICE), encoder_output, pad=decoder_pad.to(DEVICE), encoder_pad=encoder_pad.to(DEVICE))
        decoder_output = decoder_output[:, :-1, :]
        logits = self.classifier(decoder_output)

        is_special = torch.isin(label_ids.to(DEVICE), torch.tensor(SPECIAL_TOKEN_IDS, device=label_ids.device))

        mask_special_tokens = (torch.arange(self.token_count, device=logits.device) < SPECIAL_TOKEN_COUNT)
        mask_non_special_tokens = (torch.arange(self.token_count, device=logits.device) >= SPECIAL_TOKEN_COUNT)

        mask_special_tokens = mask_special_tokens.unsqueeze(0).unsqueeze(0)
        mask_non_special_tokens = mask_non_special_tokens.unsqueeze(0).unsqueeze(0)

        mask = torch.where(is_special.unsqueeze(-1), mask_special_tokens, mask_non_special_tokens)

        logits = logits.masked_fill(~mask, float('-inf'))

        return logits

    def generate(self, encoder_input_ids, max_length=MAX_SEQ_LEN):
        encoder_input_ids = encoder_input_ids.to(DEVICE)
        
        encoder_output = self.encoder(encoder_input_ids)
        batch_size = encoder_input_ids.shape[0]
        
        decoder_input_ids = torch.full((batch_size, 1), START_ID, dtype=torch.long, device=DEVICE)
        
        for _ in range(max_length - 1):
            decoder_output = self.decoder(decoder_input_ids, encoder_output)
            next_token_logits = self.classifier(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(1)], dim=1)
            
            if (next_token == END_ID).all():
                break
        
        return decoder_input_ids

if __name__ == '__main__':
    model = PixelTransformer().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())    
    memory_size = sum(p.numel() * p.element_size() for p in model.parameters())    
    print(f"Number of parameters: {num_params:,}")
    print(f"Memory size: {memory_size / (1024 * 1024):.2f} MB")
    
    raise
    from dataloader.loader import train_loader, val_loader
    from dataloader.loader import to_tokens
    for batch in val_loader:
        encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = [t.to(DEVICE) for t in batch]
        generated_ids = model.generate(encoder_input_ids[0].unsqueeze(0))
        print(to_tokens([int(id) for id in generated_ids[0].cpu().tolist()]))
        break