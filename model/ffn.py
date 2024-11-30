import torch.nn as nn
from constants import MODEL_DIM, FFN_RATIO, DROPOUT

class FFNLayer(nn.Module):
    def __init__(self, model_dimension=MODEL_DIM, ffn_ratio=FFN_RATIO) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.W_in = nn.Linear(model_dimension, ffn_ratio * model_dimension)
        self.V_in = nn.Linear(model_dimension, ffn_ratio * model_dimension)
        self.W_out = nn.Linear(ffn_ratio * model_dimension, model_dimension)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(DROPOUT)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.W_in.weight, std=0.03)
        nn.init.trunc_normal_(self.V_in.weight, std=0.03)
        nn.init.trunc_normal_(self.W_out.weight, std=0.03)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.V_in.bias)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x):
        x_skip = x.clone()
        x = self.norm(x)
        x = self.activation(self.W_in(x)) * self.V_in(x)
        x = self.dropout(x)
        x = self.W_out(x)
        x = self.dropout(x)
        return x_skip + x