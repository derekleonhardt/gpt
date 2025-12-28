import torch

# Input embeddings: each row is a token embedding vector (e.g., word embeddings)
# In attention, these represent the input sequence tokens as dense vectors
# Shape: (sequence_length, embedding_dim) = (6, 3)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1) - first token's embedding
    [0.55, 0.87, 0.66], # journey (x^2) - second token's embedding
    [0.57, 0.85, 0.64], # starts (x^3) - third token's embedding
    [0.22, 0.58, 0.33], # with (x^4) - fourth token's embedding
    [0.77, 0.25, 0.10], # one (x^5) - fifth token's embedding
    [0.05, 0.80, 0.55]] # step (x^6) - sixth token's embedding
)

attention_scores = inputs @ inputs.T
attention_weights = torch.softmax(attention_scores, dim=-1)
context_vectors = attention_weights @ inputs

print(context_vectors)

#self attention with trainable weights - scaled dot product attention

x_2 = inputs[1] #second input element
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vectors_2 = attn_weights_2 @ values

print(context_vectors_2)






