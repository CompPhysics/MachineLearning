import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# Problem setup: Runge function
# ----------------------------
def runge(x):
    # x: numpy array
    return 1.0 / (1.0 + 25.0 * x**2)

# ----------------------------
# MLP: two hidden layers
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden1=64, hidden2=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.Sigmoid(),
            nn.Linear(hidden1, hidden2),
            nn.Sigmoid(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Training utilities
# ----------------------------
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

# ----------------------------
# Main
# ----------------------------
def main():
    # Reproducibility
    torch.manual_seed(7)
    np.random.seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Generate training data on [-1, 1]
    n_train = 256
    X_train = np.random.uniform(-1.0, 1.0, size=(n_train, 1)).astype(np.float32)
    y_train = runge(X_train).astype(np.float32)

    # Small validation set
    n_val = 128
    X_val = np.random.uniform(-1.0, 1.0, size=(n_val, 1)).astype(np.float32)
    y_val = runge(X_val).astype(np.float32)

    # Torch datasets/loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    # Model, loss, optimizer (Adam)
    model = MLP(1, 128, 128, 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Optional: mild cosine LR schedule for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

    # Train
    epochs = 1500
    best_val = math.inf
    best_state = None
    for ep in range(1, epochs + 1):
        tr_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 100 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:4d} | train MSE: {tr_loss:.4e} | val MSE: {val_loss:.4e}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on a dense grid for plotting
    xs = np.linspace(-1.0, 1.0, 500, dtype=np.float32).reshape(-1, 1)
    ys_true = runge(xs).reshape(-1)
    with torch.no_grad():
        yhat = model(torch.from_numpy(xs).to(device)).cpu().numpy().reshape(-1)

    # Plot
    plt.figure()
    plt.plot(xs, ys_true, label="Runge function (true)")
    plt.plot(xs, yhat, linestyle="--", label="Neural net (prediction)")
    # also scatter training points (optional)
    plt.scatter(X_train, y_train, s=10, alpha=0.3, label="Train samples")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Approximating the Runge function with a 2-hidden-layer MLP (Adam)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
