import torch, torch.nn as nn

# A simple RNN-based model

model = nn.Sequential(nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True),nn.Linear(20, 5))

#Example input: batch of 3 sequences, each of length 7, input dim 10

x = torch.randn(3, 7, 10)
output, hn = model(x)  # output shape: (3,7,20), hn shape: (2,3,20)
