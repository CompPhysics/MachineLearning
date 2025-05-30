import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import urllib.request
import os

# ----- Utility functions -----
def load_mnist(normalize=True):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        train_set, _, _ = pickle.load(f, encoding='latin1')
    X, _ = train_set
    if normalize:
        X = X.astype(np.float32)
    return X

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# ----- Layer -----
class Dense:
    def __init__(self, in_dim, out_dim, activation='sigmoid'):
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros(out_dim)
        self.activation = activation
        self.input = None
        self.z = None

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        if self.activation == 'sigmoid':
            return sigmoid(self.z)
        elif self.activation == 'linear':
            return self.z
        elif self.activation == 'relu':
            return np.maximum(0, self.z)

    def backward(self, grad_output, learning_rate):
        if self.activation == 'sigmoid':
            grad = grad_output * sigmoid_deriv(self.z)
        elif self.activation == 'relu':
            grad = grad_output * (self.z > 0).astype(float)
        else:
            grad = grad_output
        grad_W = self.input.T @ grad
        grad_b = np.sum(grad, axis=0)
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        return grad @ self.W.T

# ----- VAE -----
class VAE:
    def __init__(self, input_dim=784, hidden_dims=[256], latent_dim=2, learning_rate=0.01):
        self.encoder_layers = [Dense(input_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            self.encoder_layers.append(Dense(hidden_dims[i - 1], hidden_dims[i]))
        self.W_mu = Dense(hidden_dims[-1], latent_dim, activation='linear')
        self.W_logvar = Dense(hidden_dims[-1], latent_dim, activation='linear')

        self.decoder_layers = [Dense(latent_dim, hidden_dims[-1])]
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(Dense(hidden_dims[i], hidden_dims[i - 1]))
        self.decoder_layers.append(Dense(hidden_dims[0], input_dim, activation='sigmoid'))

        self.learning_rate = learning_rate

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer.forward(h)
        mu = self.W_mu.forward(h)
        logvar = self.W_logvar.forward(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        h = z
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h

    def loss(self, recon_x, x, mu, logvar):
        mse = np.mean((recon_x - x) ** 2)
        kl = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
        return mse + kl

    def train_step(self, x):
        # Forward
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        loss = self.loss(x_recon, x, mu, logvar)

        # Backward
        grad = 2 * (x_recon - x) / x.shape[0]
        for layer in reversed(self.decoder_layers):
            grad = layer.backward(grad, self.learning_rate)

        # Gradients for latent
        h = self.encoder_layers[-1].z
        grad_mu = (mu / x.shape[0])
        grad_logvar = 0.5 * (np.exp(logvar) - 1) / x.shape[0]

        grad_latent = grad_mu + grad_logvar
        self.W_mu.backward(grad_mu, self.learning_rate)
        self.W_logvar.backward(grad_logvar, self.learning_rate)

        for layer in reversed(self.encoder_layers):
            grad_latent = layer.backward(grad_latent, self.learning_rate)

        return loss

    def train(self, X, epochs=10, batch_size=64):
        for epoch in range(epochs):
            perm = np.random.permutation(X.shape[0])
            total_loss = 0
            for i in range(0, X.shape[0], batch_size):
                batch = X[perm[i:i+batch_size]]
                total_loss += self.train_step(batch)
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    def reconstruct(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def sample(self, n_samples=10):
        z = np.random.randn(n_samples, self.W_mu.b.shape[0])
        return self.decode(z)

# ----- Visualize -----
def plot_reconstructions(vae, X, n=10):
    recon = vae.reconstruct(X[:n])
    fig, axs = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        axs[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].reshape(28, 28), cmap='gray')
        axs[1, i].axis('off')
    axs[0, 0].set_title('Original')
    axs[1, 0].set_title('Reconstructed')
    plt.show()

def plot_generated(vae, n=10):
    samples = vae.sample(n)
    fig, axs = plt.subplots(1, n, figsize=(n, 1.5))
    for i in range(n):
        axs[i].imshow(samples[i].reshape(28, 28), cmap='gray')
        axs[i].axis('off')
    plt.suptitle('Generated Samples')
    plt.show()

# ----- Run on MNIST -----
if __name__ == "__main__":
    X = load_mnist()[:10000]
    vae = VAE(input_dim=784, hidden_dims=[128, 64], latent_dim=2, learning_rate=0.05)
    vae.train(X, epochs=10, batch_size=128)
    plot_reconstructions(vae, X)
    plot_generated(vae)
