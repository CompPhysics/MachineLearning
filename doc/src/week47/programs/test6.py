import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import runpy
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Load your RK4 integrator and generate the dataset
# -------------------------------------------------------
data = runpy.run_path("rungekutta.py")

t = np.array(data["t"])
x = np.array(data["x"])

x = x.reshape(-1, 1)          # shape: (T, 1)
T = len(x)

# -------------------------------------------------------
# 2. Build supervised learning dataset
# -------------------------------------------------------

# ---------- Task 1: one-step predictor x_t → x_{t+1} ----------
X1 = x[:-1]
Y1 = x[1:]

X1_torch = torch.tensor(X1, dtype=torch.float32)
Y1_torch = torch.tensor(Y1, dtype=torch.float32)

# ---------- Task 4: sequence predictor ----------
seq_len = 20        # length of input window
pred_len = 20       # number of future steps to predict

X4 = []
Y4 = []

for i in range(T - seq_len - pred_len):
    X4.append(x[i : i + seq_len])
    Y4.append(x[i + seq_len : i + seq_len + pred_len])

X4 = np.array(X4)     # (N, seq_len, 1)
Y4 = np.array(Y4)     # (N, pred_len, 1)

X4_torch = torch.tensor(X4, dtype=torch.float32)
Y4_torch = torch.tensor(Y4, dtype=torch.float32)

# -------------------------------------------------------
# 3. Define RNN models
# -------------------------------------------------------

class RNNOneStep(nn.Module):
    """Model 1: x_t → x_{t+1}"""
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(1))   # shape (batch, 1, hidden)
        out = out[:, -1, :]                 # last time step
        return self.fc(out)


class RNNSequence(nn.Module):
    """Model 4: Predict multiple future steps"""
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)           # out: (batch, seq_len, hidden)
        out = self.fc(out)             # (batch, seq_len, 1)
        return out


# -------------------------------------------------------
# 4. Train Model 1 (single-step predictor)
# -------------------------------------------------------

model1 = RNNOneStep()
criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=1e-3)

for epoch in range(200):
    optimizer.zero_grad()
    pred = model1(X1_torch)
    loss = criterion(pred, Y1_torch)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"One-step Epoch {epoch}, Loss: {loss.item():.6f}")

# -------------------------------------------------------
# 5. Train Model 4 (sequence predictor)
# -------------------------------------------------------

model4 = RNNSequence()
optimizer = optim.Adam(model4.parameters(), lr=1e-3)

for epoch in range(200):
    optimizer.zero_grad()
    pred = model4(X4_torch)
    loss = criterion(pred, Y4_torch)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Sequence Epoch {epoch}, Loss: {loss.item():.6f}")

# -------------------------------------------------------
# 6. Evaluate: multi-step prediction
# -------------------------------------------------------

with torch.no_grad():
    sample_input = X4_torch[10:11]      # shape (1, seq_len, 1)
    predicted_seq = model4(sample_input).numpy().squeeze()
    true_seq = Y4[10].squeeze()

plt.plot(true_seq, label="True")
plt.plot(predicted_seq, label="Predicted", linestyle="--")
plt.legend()
plt.title("Sequence prediction (20 steps ahead)")
plt.show()
