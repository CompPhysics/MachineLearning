import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================
# 1. Physics model (from rk4.py)
# ============================

# Global parameters (same idea as in rk4.py)
gamma = 0.2        # damping
Omegatilde = 0.5   # driving frequency
Ftilde = 1.0       # driving amplitude

def spring_force(v, x, t):
    """
    SpringForce from rk4.py:
    note: divided by mass => returns acceleration
    a = -2*gamma*v - x + Ftilde*cos(Omegatilde * t)
    """
    return -2.0 * gamma * v - x + Ftilde * np.cos(Omegatilde * t)


def rk4_trajectory(DeltaT=0.001, tfinal=20.0, x0=1.0, v0=0.0):
    """
    Reimplementation of RK4 integrator from rk4.py.
    Returns t, x, v arrays.
    """
    n = int(np.ceil(tfinal / DeltaT))

    t = np.zeros(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)

    x[0] = x0
    v[0] = v0

    for i in range(n - 1):
        # k1
        k1x = DeltaT * v[i]
        k1v = DeltaT * spring_force(v[i], x[i], t[i])

        # k2
        vv = v[i] + 0.5 * k1v
        xx = x[i] + 0.5 * k1x
        k2x = DeltaT * vv
        k2v = DeltaT * spring_force(vv, xx, t[i] + 0.5 * DeltaT)

        # k3
        vv = v[i] + 0.5 * k2v
        xx = x[i] + 0.5 * k2x
        k3x = DeltaT * vv
        k3v = DeltaT * spring_force(vv, xx, t[i] + 0.5 * DeltaT)

        # k4
        vv = v[i] + k3v
        xx = x[i] + k3x
        k4x = DeltaT * vv
        k4v = DeltaT * spring_force(vv, xx, t[i] + DeltaT)

        # Update
        x[i + 1] = x[i] + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        v[i + 1] = v[i] + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        t[i + 1] = t[i] + DeltaT

    return t, x, v


# =====================================
# 2. Sequence generation for RNN training
# =====================================

def create_sequences(x, seq_len):
    """
    Given a 1D array x (e.g., position as a function of time),
    create input/target sequences for next-step prediction.

    Inputs:  [x_i, x_{i+1}, ..., x_{i+seq_len-1}]
    Targets: [x_{i+1}, ..., x_{i+seq_len}]
    """
    xs = []
    ys = []
    for i in range(len(x) - seq_len):
        seq_x = x[i : i + seq_len]
        seq_y = x[i + 1 : i + seq_len + 1]  # shifted by one step
        xs.append(seq_x)
        ys.append(seq_y)

    xs = np.array(xs, dtype=np.float32)      # shape: (num_samples, seq_len)
    ys = np.array(ys, dtype=np.float32)      # shape: (num_samples, seq_len)
    # Add feature dimension (1 feature: the position x)
    xs = np.expand_dims(xs, axis=-1)         # (num_samples, seq_len, 1)
    ys = np.expand_dims(ys, axis=-1)         # (num_samples, seq_len, 1)
    return xs, ys


class OscillatorDataset(Dataset):
    def __init__(self, seq_len=50, DeltaT=0.001, tfinal=20.0, x0=1.0, v0=0.0):
        t, x, v = rk4_trajectory(DeltaT=DeltaT, tfinal=tfinal, x0=x0, v0=v0)
        self.t = t
        self.x = x
        self.v = v
        xs, ys = create_sequences(x, seq_len=seq_len)
        self.inputs = torch.from_numpy(xs)  # (N, seq_len, 1)
        self.targets = torch.from_numpy(ys) # (N, seq_len, 1)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ==============================
# 3. RNN model (LSTM-based)
# ==============================

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)   # out: (batch, seq_len, hidden_size)
        out = self.fc(out)      # (batch, seq_len, output_size)
        return out


# ==============================
# 4. Training loop
# ==============================

def train_model(
    seq_len=50,
    DeltaT=0.001,
    tfinal=20.0,
    batch_size=64,
    num_epochs=10,
    hidden_size=64,
    lr=1e-3,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = OscillatorDataset(seq_len=seq_len, DeltaT=DeltaT, tfinal=tfinal)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = RNNPredictor(input_size=1, hidden_size=hidden_size, output_size=1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    return model, dataset


# ==============================
# 5. Evaluation / visualization
# ==============================

def evaluate_and_plot(model, dataset, seq_len=50, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        # Take a single sequence from the dataset
        x_seq, y_seq = dataset[0]  # shapes: (seq_len, 1), (seq_len, 1)
        x_input = x_seq.unsqueeze(0).to(device)  # (1, seq_len, 1)

        # Model prediction (next-step for whole sequence)
        y_pred = model(x_input).cpu().numpy().squeeze(-1).squeeze(0)  # (seq_len,)

        # True target
        y_true = y_seq.numpy().squeeze(-1)  # (seq_len,)

        # Plot comparison
        plt.figure()
        plt.plot(y_true, label="True x(t+Δt)", linestyle="-")
        plt.plot(y_pred, label="Predicted x(t+Δt)", linestyle="--")
        plt.xlabel("Time step in sequence")
        plt.ylabel("Position")
        plt.legend()
        plt.title("RNN next-step prediction on oscillator trajectory")
        plt.tight_layout()
        plt.show()


# ==============================
# 6. Main
# ==============================

if __name__ == "__main__":
    # Hyperparameters can be tweaked as you like
    seq_len = 50
    DeltaT = 0.001
    tfinal = 20.0
    num_epochs = 10
    batch_size = 64
    hidden_size = 64
    lr = 1e-3

    model, dataset = train_model(
        seq_len=seq_len,
        DeltaT=DeltaT,
        tfinal=tfinal,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        lr=lr,
    )

    evaluate_and_plot(model, dataset, seq_len=seq_len)
