import torch
import math

# Example dimensions
model_dim = 512
IMG_H, IMG_W = 4, 4

# Generate 1D positional embedding for a single position
def get_1d_pos_embedding(pos, model_dim):
    pe = torch.zeros(model_dim)
    div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
    pe[0::2] = torch.sin(pos * div_term)
    pe[1::2] = torch.cos(pos * div_term)
    return pe

# Generate 2D positional embedding for a single position (y, x)
def get_2d_pos_embedding(y, x, model_dim, IMG_H, IMG_W):
    pe = torch.zeros(model_dim)
    d_model = model_dim // 2
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[0:d_model:2] = torch.sin(x * div_term)
    pe[1:d_model:2] = torch.cos(x * div_term)
    pe[d_model::2] = torch.sin(y * div_term)
    pe[d_model+1::2] = torch.cos(y * div_term)
    return pe

# Example positions
pos_1d = 1
y, x = 1, 1

# Get embeddings
embedding_1d = get_1d_pos_embedding(pos_1d, model_dim)
embedding_2d = get_2d_pos_embedding(y, x, model_dim, IMG_H, IMG_W)

# Generate a random matrix with truncated normal distribution
mean = 0.0
std = 0.03
lower_bound = -2 * std
upper_bound = 2 * std

random_matrix = torch.empty(model_dim, model_dim).normal_(mean, std)
random_matrix = torch.clamp(random_matrix, lower_bound, upper_bound)

# Multiply the random matrix by the 1D embedding
result_vector = torch.matmul(random_matrix, embedding_1d)

# Compute the dot product with the 2D embedding
dot_product = torch.dot(result_vector, embedding_2d)

# Normalize the dot product
norm_result_vector = torch.norm(result_vector)
norm_embedding_2d = torch.norm(embedding_2d)
normalized_dot_product = dot_product / (norm_result_vector * norm_embedding_2d)

print(f"1D Embedding: {embedding_1d}")
print(f"2D Embedding: {embedding_2d}")
print(f"Random Matrix: {random_matrix}")
print(f"Result Vector: {result_vector}")
print(f"Normalized Dot Product: {normalized_dot_product.item()}")
print(norm_embedding_2d)
print(norm_result_vector)