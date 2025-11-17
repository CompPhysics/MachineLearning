import tensorflow as tf
import numpy as np

# -----------------------
# 1. Hyperparameters
# -----------------------
input_size = 10        # Dimensionality of each time step
hidden_size = 20       # Number of recurrent units
num_classes = 2        # Binary classification
sequence_length = 5     # Sequence length
batch_size = 16

# -----------------------
# 2. Dummy dataset
#    X: [batch, seq, features]
#    y: [batch]
# -----------------------
X = np.random.randn(batch_size, sequence_length, input_size).astype(np.float32)
y = np.random.randint(0, num_classes, size=(batch_size,))

# -----------------------
# 3. Build simple RNN model
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
        units=hidden_size,
        activation="tanh",
        return_sequences=False,   # Only final hidden state
        input_shape=(sequence_length, input_size)
    ),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# -----------------------
# 4. Train the model
# -----------------------
history = model.fit(
    X, y,
    epochs=5,
    batch_size=batch_size,
    verbose=1
)

# -----------------------
# 5. Evaluate
# -----------------------
logits = model.predict(X)
print("Logits from model:\n", logits)


SimpleRNN is the exact TensorFlow/Keras counterpart to PyTorch’s nn.RNN.
	•	return_sequences=False makes it output only the last hidden state, which is fed to the classifier.
	•	from_logits=True matches the PyTorch CrossEntropyLoss.
