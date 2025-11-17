import numpy as np
import torch
from torch import nn, optim

# 1. Data preparation: generate a sine wave and create input-output sequences
time_steps = np.linspace(0, 100, 500)
data = np.sin(time_steps)                   # shape (500,)
seq_length = 20
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])         # sequence of length seq_length
    y.append(data[i+seq_length])           # next value to predict
X = np.array(X)                            # shape (480, seq_length)
y = np.array(y)                            # shape (480,)
# Add feature dimension (1) for the RNN input
X = X[..., None]                           # shape (480, seq_length, 1)
y = y[..., None]                           # shape (480, 1)

# Split into train/test sets (80/20 split)
train_size = int(0.8 * len(X))
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32)
X_test  = torch.tensor(X[train_size:],  dtype=torch.float32)
y_test  = torch.tensor(y[train_size:],  dtype=torch.float32)

# 2. Model definition: simple RNN followed by a linear layer
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(SimpleRNNModel, self).__init__()
        # nn.RNN for sequential data (batch_first=True expects (batch, seq_len, features))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)    # output layer for prediction

    def forward(self, x):
        out, _ = self.rnn(x)                 # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]                  # take output of last time step
        return self.fc(out)                 # linear layer to 1D output

model = SimpleRNNModel(input_size=1, hidden_size=16, num_layers=1)
print(model)  # print model summary (structure)
# Output example:
# SimpleRNNModel(
#   (rnn): RNN(1, 16, batch_first=True)
#   (fc): Linear(in_features=16, out_features=1, bias=True)
# )



# 3. Training loop: MSE loss and Adam optimizer
criterion = nn.MSELoss()                  # mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)               # forward pass
    loss = criterion(output, y_train)     # compute training loss
    loss.backward()                       # backpropagate
    optimizer.step()                      # update weights
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
# 4. Evaluation on test set
model.eval()
with torch.no_grad():
    pred = model(X_test)
    test_loss = criterion(pred, y_test)
print(f'Test Loss: {test_loss.item():.4f}')

# (Optional) View a few actual vs. predicted values
print("Actual:", y_test[:5].flatten().numpy())
print("Pred : ", pred[:5].flatten().numpy())

