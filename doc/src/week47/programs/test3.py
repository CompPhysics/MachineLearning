import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)          # out: [batch, seq, hidden]
        last_hidden = out[:, -1, :]   # take last time step
        return self.fc(last_hidden)

class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch, seq, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)

        # Unroll manually over time
        for t in range(seq):
            xt = x[:, t, :]
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)

        return self.fc(h)

class RNNAllSteps(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.rnn(x)   # out: [batch, seq, hidden]
        return out             # return ALL time steps

class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)       # out: [batch, seq, 2*hidden]
        last_hidden = out[:, -1, :] 
        return self.fc(last_hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleRNN(input_size=10, hidden_size=20, num_layers=1, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy data
X = torch.randn(32, 5, 10).to(device)
y = torch.randint(0, 2, (32,)).to(device)

# Training step
for epoch in range(5):
    model.train()
    optimizer.zero_grad()

    logits = model(X)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")
