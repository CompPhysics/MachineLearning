# RBM Class Implementation

import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        """
        Initialize the RBM with given number of visible and hidden units.
        Weights are initialized to small random values, biases to zero.
        """
        self.n_visible = n_visible   # number of visible units
        self.n_hidden = n_hidden     # number of hidden units
        self.learning_rate = learning_rate
        
        # Initialize weight matrix (dimensions: n_visible x n_hidden) with small random values
        # Using a normal distribution with mean 0 and standard deviation 0.1 for initialization
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(n_visible, n_hidden))
        # Initialize biases for visible and hidden units as zero vectors
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
    
    def _sigmoid(self, x):
        """Sigmoid activation function to convert energies to probabilities (0 to 1)."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def sample_hidden(self, visible):
        """
        Sample hidden state given a visible state.
        - visible: array-like of shape (n_visible,) with binary values (0 or 1).
        - Returns: numpy array of shape (n_hidden,) with sampled binary hidden states.
        """
        # Ensure input is a numpy array
        visible = np.array(visible)
        # Compute activations of hidden units: W^T * visible + hidden_bias
        hidden_activations = visible.dot(self.weights) + self.hidden_bias   # shape (n_hidden,)
        # Convert activations to probabilities via sigmoid
        hidden_probs = self._sigmoid(hidden_activations)
        # Sample each hidden unit (1 with probability = hidden_probs)
        hidden_states = (np.random.rand(self.n_hidden) < hidden_probs).astype(np.int_)
        return hidden_states
    
    def sample_visible(self, hidden):
        """
        Sample visible state given a hidden state.
        - hidden: array-like of shape (n_hidden,) with binary values (0 or 1).
        - Returns: numpy array of shape (n_visible,) with sampled binary visible states.
        """
        hidden = np.array(hidden)
        # Compute activations of visible units: W * hidden + visible_bias
        visible_activations = hidden.dot(self.weights.T) + self.visible_bias  # shape (n_visible,)
        # Convert activations to probabilities via sigmoid
        visible_probs = self._sigmoid(visible_activations)
        # Sample each visible unit (1 with probability = visible_probs)
        visible_states = (np.random.rand(self.n_visible) < visible_probs).astype(np.int_)
        return visible_states
    
    def train(self, data, epochs=10, batch_size=1, k=1):
        """
        Train the RBM using Contrastive Divergence (CD-k).
        
        Parameters:
        - data: array-like of shape (num_samples, n_visible), containing the training data (values 0 or 1).
        - epochs: number of epochs (full passes through the dataset) to train.
        - batch_size: number of samples per mini-batch for updates (default 1 for stochastic CD).
        - k: number of Gibbs sampling steps for CD-k (default 1, which is the standard CD-1).
        """
        data = np.array(data)
        num_samples = data.shape[0]
        batch_size = min(batch_size, num_samples)  # ensure batch_size is valid
        
        for epoch in range(epochs):
            # Shuffle the training data at the start of each epoch
            indices = np.random.permutation(num_samples)
            data_shuffled = data[indices]
            
            # Loop over batches of the training data
            for i in range(0, num_samples, batch_size):
                batch = data_shuffled[i:i+batch_size]
                
                # === Positive phase ===
                # Compute hidden probabilities for the batch (forward pass)
                pos_hidden_activations = batch.dot(self.weights) + self.hidden_bias      # shape: (batch_size, n_hidden)
                pos_hidden_probs = self._sigmoid(pos_hidden_activations)                # p(h_j=1 | v)
                # Sample hidden states from these probabilities
                pos_hidden_states = (np.random.rand(batch.shape[0], self.n_hidden) < pos_hidden_probs).astype(np.int_)
                # Compute positive associations (correlation between v and h) as batch.T * pos_hidden_probs
                pos_associations = batch.T.dot(pos_hidden_probs)  # shape: (n_visible, n_hidden)
                
                # === Negative phase (reconstruction) ===
                # Initialize the negative phase using the sampled hidden states from the positive phase
                neg_hidden_states = pos_hidden_states
                neg_hidden_probs = None
                neg_visible_states = None
                
                # Perform k full Gibbs sampling steps (hidden->visible->hidden repeated k times)
                for step in range(k):
                    # Visible probabilities given hidden
                    neg_visible_activations = neg_hidden_states.dot(self.weights.T) + self.visible_bias   # (batch_size, n_visible)
                    neg_visible_probs = self._sigmoid(neg_visible_activations)                           # p(v_i=1 | h)
                    # Sample visible states from these probabilities
                    neg_visible_states = (np.random.rand(batch.shape[0], self.n_visible) < neg_visible_probs).astype(np.int_)
                    
                    # Hidden probabilities given the reconstructed visible
                    neg_hidden_activations = neg_visible_states.dot(self.weights) + self.hidden_bias     # (batch_size, n_hidden)
                    neg_hidden_probs = self._sigmoid(neg_hidden_activations)                            # p(h_j=1 | v_recon)
                    # Sample hidden states for the next step (if more Gibbs steps remain)
                    if step < k - 1:
                        neg_hidden_states = (np.random.rand(batch.shape[0], self.n_hidden) < neg_hidden_probs).astype(np.int_)
                
                # After k steps, we have neg_visible_states (sampled reconstruction) and neg_hidden_probs (probabilities at hidden layer)
                
                # === Update weights and biases ===
                # Compute negative associations (using reconstructed visible states and hidden probabilities)
                neg_associations = neg_visible_states.T.dot(neg_hidden_probs)  # shape: (n_visible, n_hidden)
                # Update weights: increase by learning_rate * (positive associations - negative associations) normalized by batch size
                self.weights += self.learning_rate * (pos_associations - neg_associations) / batch.shape[0]
                # Update biases: adjust towards the data (positive phase) and away from reconstructions (negative phase)
                self.visible_bias += self.learning_rate * np.mean(batch - neg_visible_states, axis=0)
                self.hidden_bias  += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
            # (Optionally, one could compute reconstruction error here to monitor training progress)
    
    def reconstruct(self, visible):
        """
        Reconstruct a given visible vector through one forward-pass to hidden and one backward-pass to visible.
        - visible: array-like of shape (n_visible,) with binary values.
        - Returns: tuple (reconstructed_visible_probabilities, reconstructed_visible_state).
        """
        visible = np.array(visible)
        # Forward pass: visible -> hidden
        hidden_probs = self._sigmoid(visible.dot(self.weights) + self.hidden_bias)
        hidden_state = (np.random.rand(self.n_hidden) < hidden_probs).astype(np.int_)
        # Backward pass: hidden -> visible
        recon_visible_activations = hidden_state.dot(self.weights.T) + self.visible_bias
        recon_visible_probs = self._sigmoid(recon_visible_activations)
        recon_visible_state = (np.random.rand(self.n_visible) < recon_visible_probs).astype(np.int_)
        return recon_visible_probs, recon_visible_state
    
    def get_parameters(self):
        """
        Get the learned parameters of the RBM.
        Returns a tuple: (weight_matrix, visible_bias_vector, hidden_bias_vector).
        """
        return self.weights, self.visible_bias, self.hidden_bias
    
    def gibbs_sample(self, num_steps=1, initial_visible=None):
        """
        Generate a sample from the learned RBM by running Gibbs sampling.
        - num_steps: number of Gibbs sampling iterations to perform.
        - initial_visible: optional initial visible state to start the chain (if None, a random visible state is used).
        - Returns: tuple (visible_state, hidden_state) after the Gibbs sampling steps.
        """
        # If no initial visible state provided, start with a random binary visible vector
        if initial_visible is None:
            visible_state = (np.random.rand(self.n_visible) < 0.5).astype(np.int_)
        else:
            visible_state = np.array(initial_visible).astype(np.int_)
        
        hidden_state = None
        # Iterate Gibbs sampling for the specified number of steps
        for step in range(num_steps):
            # Sample hidden state given current visible state
            hidden_probs = self._sigmoid(visible_state.dot(self.weights) + self.hidden_bias)
            hidden_state = (np.random.rand(self.n_hidden) < hidden_probs).astype(np.int_)
            # Sample visible state given current hidden state
            visible_probs = self._sigmoid(hidden_state.dot(self.weights.T) + self.visible_bias)
            visible_state = (np.random.rand(self.n_visible) < visible_probs).astype(np.int_)
        # After the final step, we have a new visible_state; sample a final hidden_state for completeness
        hidden_probs = self._sigmoid(visible_state.dot(self.weights) + self.hidden_bias)
        hidden_state = (np.random.rand(self.n_hidden) < hidden_probs).astype(np.int_)
        return visible_state, hidden_state

    

# Example usage of the RBM class

# 1. Create a synthetic dataset (binary patterns)
#    Here we use patterns of length 4. For example: 
#    Pattern A = [1, 1, 0, 0] and Pattern B = [0, 0, 1, 1], repeated multiple times.
data = [[1, 1, 0, 0]] * 50 + [[0, 0, 1, 1]] * 50  # 100 samples total (50 of each pattern)

# 2. Initialize the RBM with 4 visible units and 2 hidden units
rbm = RBM(n_visible=4, n_hidden=2, learning_rate=0.1)

# 3. Train the RBM on the dataset for a certain number of epochs
rbm.train(data, epochs=50, batch_size=10, k=1)  # using CD-1 (k=1) and mini-batches of size 10

# 4. After training, retrieve the learned weights and biases
weights, vis_bias, hid_bias = rbm.get_parameters()
print("Learned weight matrix:\n", weights)
print("Learned visible biases:", vis_bias)
print("Learned hidden biases:", hid_bias)

# 5. Reconstruct an example from the training data using the trained RBM
test_visible = np.array([1, 1, 0, 0])  # input pattern A
recon_probs, recon_state = rbm.reconstruct(test_visible)
print("\nOriginal visible input:", test_visible)
print("Reconstructed visible probabilities:", recon_probs)
print("Reconstructed visible sample:", recon_state)

# 6. Generate a new sample from the model using Gibbs sampling
visible_sample, hidden_sample = rbm.gibbs_sample(num_steps=5)  # start from a random visible state
print("\nNew sampled visible state (after Gibbs sampling):", visible_sample)
print("Corresponding hidden state:", hidden_sample)


