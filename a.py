import torch
from torch import nn
device = torch.device('cuda')
embedding = nn.Embedding(230, 256).to(device)
input = torch.LongTensor([[0.1, 0.2, 0.4, 0.5], [0.4, 0.3, 0.2, 0.9]]).to(device)

embedding(input)
print('aaa')