import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle, os, urllib.request

# ===== Utility functions =====
def load_mnist():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    fname = 'mnist.pkl.gz'
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, 'rb') as f:
        train_set, _, _ = pickle.load(f, encoding='latin1')
    X, _ = train_set
    return X.astype(np.float32)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)

# ===== Neural network for epsilon_theta =====
class Dense:
    def __init__(self, in_dim, out_dim, activation='relu'):
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros(out_dim)
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            self.out = relu(self.z)
        elif self.activation == 'linear':
            self.out = self.z
        return self.out

    def backward(self, grad_out, lr):
        if self.activation == 'relu':
            grad = grad_out * (self.z > 0).astype(float)
        else:
            grad = grad_out

        dW = self.input.T @ grad
        db = np.sum(grad, axis=0)
        self.W -= lr * dW
        self.b -= lr * db
        return grad @ self.W.T

class DenoiseMLP:
    def __init__(self, input_dim, hidden_dims):
        dims = [input_dim] + hidden_dims + [input_dim]
        self.layers = [Dense(dims[i], dims[i+1], 'relu' if i < len(dims)-2 else 'linear') for i in range(len(dims)-1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

# ===== Variational Diffusion Model =====
class DiffusionModel:
    def __init__(self, img_dim, timesteps=1000, hidden_dims=[512, 256], lr=1e-3):
        self.T = timesteps
        self.beta = linear_beta_schedule(self.T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

        self.model = DenoiseMLP(input_dim=img_dim, hidden_dims=hidden_dims)
        self.lr = lr
        self.img_dim = img_dim

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = np.random.randn(*x0.shape)
        sqrt_alpha_bar = np.sqrt(self.alpha_bar[t])[:, None]
        sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar[t])[:, None]
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def train_step(self, x):
        N = x.shape[0]
        t = np.random.randint(0, self.T, size=N)
        noise = np.random.randn(*x.shape)
        xt = self.q_sample(x, t, noise)
        pred_noise = self.model.forward(xt)

        loss = np.mean((pred_noise - noise) ** 2)
        grad = 2 * (pred_noise - noise) / N
        self.model.backward(grad, self.lr)
        return loss

    def train(self, data, epochs=10, batch_size=128):
        for epoch in range(epochs):
            perm = np.random.permutation(len(data))
            total_loss = 0
            for i in range(0, len(data), batch_size):
                x = data[perm[i:i+batch_size]]
                total_loss += self.train_step(x)
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    def p_sample(self, xt, t):
        pred_noise = self.model.forward(xt)
        alpha = self.alpha[t]
        alpha_bar = self.alpha_bar[t]
        beta = self.beta[t]

        coef1 = 1 / np.sqrt(alpha)
        coef2 = (1 - alpha) / np.sqrt(1 - alpha_bar)
        mean = coef1 * (xt - coef2 * pred_noise)

        if t > 0:
            noise = np.random.randn(*xt.shape)
        else:
            noise = 0
        return mean + np.sqrt(beta) * noise

    def sample(self, n=16):
        xt = np.random.randn(n, self.img_dim)
        for t in reversed(range(self.T)):
            xt = self.p_sample(xt, t)
        return xt

# ===== Visualization =====
def plot_images(samples, n=8):
    fig, axs = plt.subplots(1, n, figsize=(n, 1.5))
    for i in range(n):
        axs[i].imshow(samples[i].reshape(28, 28), cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Generated Samples")
    plt.show()

# ===== Run full example =====
if __name__ == "__main__":
    X = load_mnist()[:5000]
    model = DiffusionModel(img_dim=784, timesteps=100, hidden_dims=[256, 128], lr=1e-3)
    model.train(X, epochs=10, batch_size=128)
    samples = model.sample(n=8)
    plot_images(samples)
