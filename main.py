import torch
from torch.utils.data import DataLoader


def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot

y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc)



Z = torch.tensor( [[-0.3,  -0.5, -0.5],
                   [-0.4,  -0.1, -0.5],
                   [-0.3,  -0.94, -0.5],
                   [-0.99, -0.88, -0.5]])

print('Z:\n', Z)

def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)). t()

smax = softmax(Z)
print('softmax:\n', smax)



# initiating data loader


