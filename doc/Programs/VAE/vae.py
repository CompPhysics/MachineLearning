import numpy as np

class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Encoder weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights and biases
        self.W2 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(hidden_dim)

        self.W_out = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_out = np.zeros(input_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def binary_cross_entropy(self, recon_x, x):
        eps = 1e-8
        return -np.sum(x * np.log(recon_x + eps) + (1 - x) * np.log(1 - recon_x + eps))

    def encode(self, x):
        h = self.sigmoid(np.dot(x, self.W1) + self.b1)
        mu = np.dot(h, self.W_mu) + self.b_mu
        logvar = np.dot(h, self.W_logvar) + self.b_logvar
        return mu, logvar, h

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        h_dec = self.sigmoid(np.dot(z, self.W2) + self.b2)
        x_recon = self.sigmoid(np.dot(h_dec, self.W_out) + self.b_out)
        return x_recon, h_dec

    def compute_loss(self, x, x_recon, mu, logvar):
        bce = self.binary_cross_entropy(x_recon, x)
        kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        return bce + kl

    def train(self, data, epochs=100, batch_size=10):
        data = np.array(data)
        n_samples = data.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                batch = data[indices[i:i + batch_size]]
                grads = self._compute_gradients(batch)
                self._update_parameters(grads)
                total_loss += grads['loss']

            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def _compute_gradients(self, x):
        m = x.shape[0]

        # Forward pass
        mu, logvar, h_enc = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon, h_dec = self.decode(z)

        # Loss
        loss = self.compute_loss(x, x_recon, mu, logvar)

        # Backpropagation (simplified SGD)
        # Output layer
        delta_out = x_recon - x  # (m, input_dim)
        dW_out = np.dot(h_dec.T, delta_out) / m
        db_out = np.mean(delta_out, axis=0)

        # Decoder hidden layer
        delta_dec = np.dot(delta_out, self.W_out.T) * self.sigmoid_derivative(np.dot(z, self.W2) + self.b2)
        dW2 = np.dot(z.T, delta_dec) / m
        db2 = np.mean(delta_dec, axis=0)

        # Latent space gradients
        dz = np.dot(delta_dec, self.W2.T)

        dmu = dz + mu / m
        dlogvar = 0.5 * dz * (np.exp(0.5 * logvar)) / m

        # Encoder hidden layer
        dh = (np.dot(dmu, self.W_mu.T) + np.dot(dlogvar, self.W_logvar.T)) * self.sigmoid_derivative(np.dot(x, self.W1) + self.b1)

        dW_mu = np.dot(h_enc.T, dmu) / m
        db_mu = np.mean(dmu, axis=0)

        dW_logvar = np.dot(h_enc.T, dlogvar) / m
        db_logvar = np.mean(dlogvar, axis=0)

        dW1 = np.dot(x.T, dh) / m
        db1 = np.mean(dh, axis=0)

        return {
            'dW1': dW1, 'db1': db1,
            'dW_mu': dW_mu, 'db_mu': db_mu,
            'dW_logvar': dW_logvar, 'db_logvar': db_logvar,
            'dW2': dW2, 'db2': db2,
            'dW_out': dW_out, 'db_out': db_out,
            'loss': loss
        }

    def _update_parameters(self, grads):
        self.W1 -= self.learning_rate * grads['dW1']
        self.b1 -= self.learning_rate * grads['db1']

        self.W_mu -= self.learning_rate * grads['dW_mu']
        self.b_mu -= self.learning_rate * grads['db_mu']

        self.W_logvar -= self.learning_rate * grads['dW_logvar']
        self.b_logvar -= self.learning_rate * grads['db_logvar']

        self.W2 -= self.learning_rate * grads['dW2']
        self.b2 -= self.learning_rate * grads['db2']

        self.W_out -= self.learning_rate * grads['dW_out']
        self.b_out -= self.learning_rate * grads['db_out']

    def reconstruct(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon, _ = self.decode(z)
        return x_recon

    def sample(self, n_samples=1):
        z = np.random.randn(n_samples, self.latent_dim)
        x_recon, _ = self.decode(z)
        return x_recon


# Create synthetic binary data (patterns)
np.random.seed(42)
data = np.random.binomial(n=1, p=0.5, size=(100, 6))

# Initialize VAE: input=6, hidden=4, latent=2
vae = VAE(input_dim=6, hidden_dim=4, latent_dim=2, learning_rate=0.1)

# Train
vae.train(data, epochs=50, batch_size=10)

# Reconstruct
x_test = data[0]
x_recon = vae.reconstruct(x_test.reshape(1, -1))
print("\nOriginal:    ", x_test)
print("Reconstructed:", np.round(x_recon[0], 2))

# Generate new samples
samples = vae.sample(n_samples=3)
print("\nGenerated samples:")
print(np.round(samples, 2))

