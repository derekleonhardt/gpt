import torch
from torch.utils.data import DataLoader

# initiating data loader
import os
import importlib.util

# Import bpe-tokenizer (handles hyphen in filename)
script_dir = os.path.dirname(os.path.abspath(__file__))
bpe_tokenizer_path = os.path.join(script_dir, 'bpe-tokenizer.py')
spec = importlib.util.spec_from_file_location("bpe_tokenizer", bpe_tokenizer_path)

bpe_tokenizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpe_tokenizer)
create_dataloader = bpe_tokenizer.create_dataloader

# Get the path to the-verdict.txt in the parent directory
parent_dir = os.path.dirname(script_dir)
file_path = os.path.join(parent_dir, 'the-verdict.txt')

with open(file_path, 'r') as file:
    raw_text = file.read()

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
