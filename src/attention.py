import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vectors = attention_weights @ values
        
        return context_vectors

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vectors = attention_weights @ values

        return context_vectors

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1) - first token's embedding
    [0.55, 0.87, 0.66], # journey (x^2) - second token's embedding
    [0.57, 0.85, 0.64], # starts (x^3) - third token's embedding
    [0.22, 0.58, 0.33], # with (x^4) - fourth token's embedding
    [0.77, 0.25, 0.10], # one (x^5) - fifth token's embedding
    [0.05, 0.80, 0.55]] # step (x^6) - sixth token's embedding
)
torch.manual_seed(123)
d_in, d_out = 3, 2
sa_v1 = SelfAttention_v1(d_in, d_out)
output = sa_v1(inputs)
print("Output shape:", output.shape)
print("Output:", output)

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
output_v2 = sa_v2(inputs)
print("Output shape (v2):", output_v2.shape)
print("Output (v2):", output_v2)



sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v2 = SelfAttention_v2(d_in, d_out)

# Copy weights from v2 to v1
with torch.no_grad():  # Disable gradient tracking during weight copy
    sa_v1.W_query.data = sa_v2.W_query.weight.T.clone()
    sa_v1.W_key.data = sa_v2.W_key.weight.T.clone()
    sa_v1.W_value.data = sa_v2.W_value.weight.T.clone()

# Get outputs
output_v1 = sa_v1(inputs)
output_v2 = sa_v2(inputs)

# Verify they match
print("v1 output:", output_v1)
print("v2 output:", output_v2)
print("Outputs match:", torch.allclose(output_v1, output_v2, atol=1e-6))