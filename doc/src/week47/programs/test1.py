import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1. Hyperparameters
# -----------------------
input_size = 10
hidden_size = 20
num_layers = 1
num_classes = 2
sequence_length = 5
batch_size = 16
lr = 1e-3

# -----------------------
# 2. Dummy dataset
# -----------------------
X = torch.randn(batch_size, sequence_length, input_size)
y = torch.randint(0, num_classes, (batch_size,))

# -----------------------
# 3. Simple RNN model
# -----------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, h_n = self.rnn(x)   # out: [batch, seq, hidden]

        # ---- FIX: take only the last time-step tensor ----
        last_hidden = out[:, -1, :]  # [batch, hidden]

        logits = self.fc(last_hidden)
        return logits

model = SimpleRNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# 4. Training step
# -----------------------
model.train()
optimizer.zero_grad()

logits = model(X)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
