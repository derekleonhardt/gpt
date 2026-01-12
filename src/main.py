import torch
from model import GPTModel, GPT_CONFIG_124M

torch.manual_seed(123)
model = gptModel(GPT_CONFIG_124M)
model.eval()


