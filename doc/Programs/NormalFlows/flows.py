import numpy as np

class CouplingLayer:
    def __init__(self, input_dim, mask, hidden_dim=32):
        """
        Affine coupling layer for RealNVP.
        - input_dim: int, dimension of input vectors.
        - mask: binary 0/1 array of length input_dim, 
                1 for indices to remain unchanged, 0 for indices to transform.
        - hidden_dim: int, number of hidden units in the scale/shift network.
        """
        self.D = input_dim
        # Store mask and pre-compute index sets for efficiency
        self.mask = mask.astype(np.float32)
        self.mask_indices1 = np.where(self.mask == 1)[0]  # indices of unchanged (masked) features
        self.mask_indices0 = np.where(self.mask == 0)[0]  # indices of transformed features
        self.d1 = len(self.mask_indices1)  # number of masked features
        self.d0 = len(self.mask_indices0)  # number of transformed features

        # Initialize weights for the coupling neural network (1 hidden layer MLP)
        output_dim = 2 * self.d0  # outputs: [t,...,t, s,...,s] for each transformed feature
        # Weight initialization with small random values for stability
        self.W1 = 0.01 * np.random.randn(self.d1, hidden_dim).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = 0.01 * np.random.randn(hidden_dim, output_dim).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        # Placeholders for gradients (same shapes as weights)
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

        # Cache variables for forward pass (needed for backpropagation)
        self.x1 = None  # masked input part
        self.x0 = None  # transformed input part
        self.h_pre = None  # pre-activation of hidden layer
        self.h = None     # post-activation (ReLU) hidden output
        self.s = None     # scale outputs (log scale)
        self.t = None     # shift outputs

    def forward(self, x):
        """
        Forward transform: given input x, compute output z and log-determinant.
        x: array of shape (N, input_dim) for N data points.
        Returns: z (N, input_dim) and log_det (N,) for this layer.
        """
        x = x.astype(np.float32)
        # Split x into masked (pass-through) and transformed parts
        x1 = x[:, self.mask_indices1]  # shape (N, d1)
        x0 = x[:, self.mask_indices0]  # shape (N, d0)
        # Cache for backward
        self.x1, self.x0 = x1, x0

        # Compute scale and shift via the neural network
        # Hidden layer: ReLU(W1 * x1 + b1)
        self.h_pre = x1.dot(self.W1) + self.b1            # (N, hidden_dim)
        self.h = np.maximum(self.h_pre, 0.0)              # apply ReLU
        out = self.h.dot(self.W2) + self.b2               # (N, 2*d0) output

        # Split output into shift (t) and log-scale (s)
        self.t = out[:, :self.d0]        # first half for translations
        self.s = out[:, self.d0:]        # second half for log scale factors

        # Affine transform: z0 = (x0 - t) * exp(-s), and z1 = x1 (identity for masked part)
        z0 = (x0 - self.t) * np.exp(-self.s)
        z1 = x1  # unchanged part
        # Combine z0 and z1 back into full output z
        z = np.empty_like(x)
        z[:, self.mask_indices1] = z1
        z[:, self.mask_indices0] = z0

        # Log-determinant of Jacobian for this layer: 
        # For each transformed feature: ∂z0/∂x0 = exp(-s), so log|det| = -sum(s)
        log_det = -np.sum(self.s, axis=1)
        return z, log_det

    def inverse(self, z):
        """
        Inverse transform: given z, compute x (applies the affine coupling in reverse).
        z: array of shape (N, input_dim).
        Returns: x of shape (N, input_dim).
        """
        z = z.astype(np.float32)
        # Split z into masked and transformed parts
        z1 = z[:, self.mask_indices1]  # unchanged part
        z0 = z[:, self.mask_indices0]
        # Compute scale and shift from z1 (same network as forward)
        h_pre = z1.dot(self.W1) + self.b1
        h = np.maximum(h_pre, 0.0)
        out = h.dot(self.W2) + self.b2
        t = out[:, :self.d0]
        s = out[:, self.d0:]
        # Inverse affine: x0 = z0 * exp(s) + t, x1 = z1
        x0 = z0 * np.exp(s) + t
        x1 = z1
        # Combine into full x
        x = np.empty_like(z)
        x[:, self.mask_indices1] = x1
        x[:, self.mask_indices0] = x0
        return x

    def backward(self, grad_output):
        """
        Backward pass: compute gradients of loss w.rt this layer's parameters and input.
        grad_output: array (N, input_dim) = ∂L/∂z from the next layer (or base log-prob).
        Returns: grad_input = ∂L/∂x, with shape (N, input_dim).
        """
        # Split gradient from output into parts corresponding to z1 and z0
        grad_z1 = grad_output[:, self.mask_indices1]  # (N, d1)
        grad_z0 = grad_output[:, self.mask_indices0]  # (N, d0)

        # Use chain rule through z0 = (x0 - t) * exp(-s) and z1 = x1:
        # Let y = x0 - t. Then z0 = y * exp(-s).
        # Compute gradients ∂L/∂y, ∂L/∂s, ∂L/∂t:
        grad_y = grad_z0 * np.exp(-self.s)                     # ∂L/∂y = ∂L/∂z0 * exp(-s)
        grad_s_out = grad_z0 * (-(self.x0 - self.t) * np.exp(-self.s))  # ∂L/∂s = ∂L/∂z0 * (-y * exp(-s))
        grad_t_out = -grad_y                                   # ∂L/∂t = ∂L/∂y * (-1)

        # Now z1 = x1 (identity), so ∂L/∂x1 gets contributions:
        grad_x1 = grad_z1                     # direct from identity path
        # (plus we will add indirect contributions from t and s via network input)

        # Combine grad_t and grad_s for network output (concatenate along feature dim)
        grad_out = np.concatenate([grad_t_out, grad_s_out], axis=1)  # shape (N, 2*d0)

        # Backprop through output linear layer: out = h.dot(W2) + b2
        # Gradients for W2 and b2 (summing over batch):
        self.grad_W2 = self.h.T.dot(grad_out)       # shape (hidden_dim, 2*d0)
        self.grad_b2 = grad_out.sum(axis=0)         # shape (2*d0,)
        # Gradient w.rt hidden layer activations h
        grad_h = grad_out.dot(self.W2.T)            # shape (N, hidden_dim)

        # Backprop through ReLU: h = max(h_pre, 0)
        grad_h_pre = grad_h.copy()
        grad_h_pre[self.h_pre <= 0] = 0.0           # gradient is zero where ReLU was inactive

        # Gradients for W1 and b1
        self.grad_W1 = self.x1.T.dot(grad_h_pre)    # shape (d1, hidden_dim)
        self.grad_b1 = grad_h_pre.sum(axis=0)       # shape (hidden_dim,)

        # Gradient w.rt input x1 (masked part) via the network path
        grad_x1_network = grad_h_pre.dot(self.W1.T) # (N, d1)
        # Total gradient for x1 combines direct identity and network paths:
        grad_x1_total = grad_x1 + grad_x1_network

        # Gradient w.rt input x0 (transformed part) is grad_y (since ∂y/∂x0 = 1)
        grad_x0 = grad_y

        # Reassemble gradient for full input x
        grad_input = np.empty_like(grad_output)
        grad_input[:, self.mask_indices1] = grad_x1_total
        grad_input[:, self.mask_indices0] = grad_x0
        return grad_input

class RealNVP:
    def __init__(self, input_dim, n_coupling_layers=4, hidden_dim=32):
        """
        RealNVP flow composed of multiple coupling layers.
        - input_dim: dimension of the input data.
        - n_coupling_layers: number of coupling layers to stack.
        - hidden_dim: hidden layer size for each coupling layer's network.
        """
        self.D = input_dim
        self.layers = []
        # Define alternating masks (here we use half-half masking)
        d = input_dim // 2
        mask1 = np.array([1]*d + [0]*(input_dim - d), dtype=np.float32)  # first half pass-through
        mask2 = 1 - mask1                                               # second half pass-through
        masks = [mask1, mask2]
        # Create coupling layers with alternating masks
        for i in range(n_coupling_layers):
            mask = masks[i % 2]
            self.layers.append(CouplingLayer(input_dim, mask, hidden_dim=hidden_dim))

    def forward(self, x):
        """
        Forward pass through all coupling layers: x (data) -> z (latent).
        Returns: z (N, D) and total log_det (N,) for the whole flow.
        """
        x = x.astype(np.float32)
        log_det_total = np.zeros(x.shape[0], dtype=np.float32)
        z = x
        for layer in self.layers:
            z, log_det = layer.forward(z)
            log_det_total += log_det       # accumulate log-determinants from each layer
        return z, log_det_total

    def inverse(self, z):
        """
        Inverse pass: z (latent) -> x (data) by inverting all coupling layers.
        """
        x = z.astype(np.float32)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """
        Compute log-likelihood of data points x under the flow model.
        """
        z, log_det_total = self.forward(x)
        # log probability of z under base (standard Gaussian N(0,I))
        log_pz = -0.5 * np.sum(z**2, axis=1) - 0.5 * self.D * np.log(2*np.pi)
        # total log-prob = base log-prob + log-determinant of transform
        return log_pz + log_det_total

    def sample(self, n):
        """
        Draw n samples from the flow model (by sampling base z ~ N(0,I) and transforming to x).
        """
        z = np.random.normal(size=(n, self.D)).astype(np.float32)
        x = self.inverse(z)
        return x

    def train(self, data, lr=1e-3, epochs=100, batch_size=100, verbose=True):
        """
        Train the RealNVP model on given data using gradient descent.
        - data: array of shape (N, D)
        - lr: learning rate
        - epochs: number of passes over the data
        - batch_size: mini-batch size for stochastic gradient descent
        - verbose: if True, print loss every 10% of training or so.
        """
        N = data.shape[0]
        for epoch in range(epochs):
            # Shuffle data for this epoch
            perm = np.random.permutation(N)
            data_shuffled = data[perm]
            avg_loss = 0.0
            # Mini-batch training
            for i in range(0, N, batch_size):
                batch = data_shuffled[i:i+batch_size]
                # Forward pass: compute log-likelihood and loss
                z, log_det = self.forward(batch)
                log_pz = -0.5 * np.sum(z**2, axis=1) - 0.5 * self.D * np.log(2*np.pi)
                log_likelihood = log_pz + log_det        # log p(x) for each sample in batch
                loss = -np.mean(log_likelihood)          # negative log-likelihood to minimize
                # Compute gradient of loss w.rt z (using dL/dz = z for standard normal base)
                grad_z = z  # since ∂(-log p_base)/∂z = z for N(0,I)
                # Backpropagate through all layers
                grad = grad_z
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                # Gradient descent: update each layer's parameters
                for layer in self.layers:
                    layer.W1 -= lr * (layer.grad_W1 / batch.shape[0])  # use batch avg gradients
                    layer.b1 -= lr * (layer.grad_b1 / batch.shape[0])
                    layer.W2 -= lr * (layer.grad_W2 / batch.shape[0])
                    layer.b2 -= lr * (layer.grad_b2 / batch.shape[0])
                avg_loss += loss * batch.shape[0]
            avg_loss /= N
            if verbose and (epoch % max(1, epochs//10) == 0):
                print(f"Epoch {epoch+1}/{epochs}, Negative Log-Lik: {avg_loss:.4f}")


import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic 2D data (two Gaussian blobs)
data, _ = make_blobs(n_samples=1000, centers=[(-2,0), (2,0)], cluster_std=0.5, random_state=42)
data = data.astype(np.float32)

# Initialize RealNVP model
flow = RealNVP(input_dim=2, n_coupling_layers=6, hidden_dim=64)

# Train the model on the data
losses = flow.train(data, lr=5e-4, epochs=300, batch_size=100, verbose=False)
print(f"Final training negative log-likelihood: {losses[-1]:.4f}")

# Generate some samples from the trained flow
samples = flow.sample(500)
