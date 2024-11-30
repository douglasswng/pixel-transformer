import torch
import torch.nn as nn
from constants import MODEL_DIM, MAX_SEQ_LEN, ATTENTION_HEADS, DROPOUT

class RotaryPostionalEmbedding(nn.Module):
    def __init__(self, model_dimension: int = MODEL_DIM, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        theta = torch.tensor([10000 ** (-2 * i / model_dimension) for i in range(model_dimension // 2)])
        m_theta = torch.stack([m * theta for m in range(1, 2 * max_seq_len - 2)])       
        m_theta = m_theta.repeat_interleave(2, dim=-1)
        self.register_buffer('cos', torch.cos(m_theta))
        self.register_buffer('sin', torch.sin(m_theta))

    def shuffle(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2) // 2, 2)
        x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        x = x.view(x.size(0), x.size(1), x.size(2) * 2)
        return x
    
    def forward(self, x):
        token_count = x.size(-2)
        x = x * self.cos[:token_count].unsqueeze(0) + self.shuffle(x) * self.sin[:token_count].unsqueeze(0)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension=MODEL_DIM, attention_heads=ATTENTION_HEADS):
        super().__init__()
        self.model_dimension = model_dimension
        self.attention_heads = attention_heads
        self.head_dimension = model_dimension // attention_heads
        
        self.pos_embedding = RotaryPostionalEmbedding()
        self.W_q = nn.Linear(model_dimension, model_dimension, bias=False)
        self.W_k = nn.Linear(model_dimension, model_dimension, bias=False)
        self.W_v = nn.Linear(model_dimension, model_dimension, bias=False)
        self.W_o = nn.Linear(model_dimension, model_dimension, bias=False)

        self.dropout = nn.Dropout(DROPOUT)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02)

    def attention_scores(self, Q_matrix, K_matrix, causal_mask=False):
        Q_matrix = Q_matrix.view(*Q_matrix.shape[:-1], self.attention_heads, self.head_dimension)
        K_matrix = K_matrix.view(*K_matrix.shape[:-1], self.attention_heads, self.head_dimension)

        dot_products = torch.einsum('bthd, bshd -> bhts', Q_matrix, K_matrix)
        
        scaling = torch.sqrt(torch.tensor(self.head_dimension, device=Q_matrix.device))
        scaled_dots = dot_products / scaling

        if causal_mask:
            seq_len = Q_matrix.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q_matrix.device), diagonal=1).bool()
            scaled_dots.masked_fill_(mask, float('-inf'))

        attention_scores = torch.softmax(scaled_dots, dim=-1)
        return attention_scores
    
    def attention(self, attention_scores, V_matrix):
        V_matrix = V_matrix.view(*V_matrix.shape[:-1], self.attention_heads, self.head_dimension)

        attention = torch.einsum('bhts, bshd -> bthd', attention_scores, V_matrix)
        attention = attention.contiguous().view(*attention.shape[:-2], self.model_dimension)
        return attention

    def self_attention(self, x, pad=None, causal_mask=False):
        if pad is not None:
            pad = pad.unsqueeze(-1)
            x = x.masked_fill(pad, 0)

        Q_matrix = self.W_q(x)
        K_matrix = self.W_k(x)
        V_matrix = self.W_v(x)

        Q_matrix = self.pos_embedding(Q_matrix)
        K_matrix = self.pos_embedding(K_matrix)

        attention_scores = self.attention_scores(Q_matrix, K_matrix, causal_mask)
        attention_scores = self.dropout(attention_scores)

        attention = self.attention(attention_scores, V_matrix)

        output = self.W_o(attention)

        return output

    def cross_attention(self, x, context, pad=None, context_pad=None):
        if pad is not None:
            pad = pad.unsqueeze(-1)
            x = x.masked_fill(pad, 0)
        
        if context_pad is not None:
            context_pad = context_pad.unsqueeze(-1)
            context = context.masked_fill(context_pad, 0)

        Q_matrix = self.W_q(x)
        K_matrix = self.W_k(context)
        V_matrix = self.W_v(context)

        Q_matrix = self.pos_embedding(Q_matrix)

        attention_scores = self.attention_scores(Q_matrix, K_matrix)
        attention_scores = self.dropout(attention_scores)

        attention = self.attention(attention_scores, V_matrix)

        output = self.W_o(attention)

        return output

    def forward(self, x, context=None, pad=None, context_pad=None, causal_mask=False):
        if context is None:
            return self.self_attention(x, pad, causal_mask)
        else:
            return self.cross_attention(x, context, pad, context_pad)
