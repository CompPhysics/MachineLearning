# train_rnn_from_rk4.py
# Usage: put this file in the same folder as your rungekutta.py and run:
#   python train_rnn_from_rk4.py

import runpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# ---------- Load RK4-generated data from your script ----------
# This runs rungekutta.py and collects its globals. It must populate 't' and 'x' arrays.
g = runpy.run_path('rungekutta.py')

if not all(k in g for k in ('t','x','v')):
    raise RuntimeError("rungekutta.py did not expose required variables 't', 'x', 'v' in its globals.")

t = np.array(g['t']).ravel()
x = np.array(g['x']).ravel()
v = np.array(g['v']).ravel()

print("Loaded shapes:", t.shape, x.shape, v.shape)

# Simple plot of the original trajectory
plt.figure(figsize=(8,3))
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('True trajectory from RK4')
plt.tight_layout()
plt.show()

# ---------- Prepare datasets ----------
def make_dataset(series, input_len):
    X, y = [], []
    N = len(series)
    for i in range(N - input_len):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len])
    X = np.array(X).reshape(-1, input_len, 1)  # (samples, timesteps, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

# normalize using global mean/std
mean_x, std_x = x.mean(), x.std()
x_norm = (x - mean_x) / std_x

print(f"Normalization: mean={mean_x:.6f}, std={std_x:.6f}")

# Model A: input_len = 1 (x_t -> x_{t+1})
input_len_A = 1
X_A, y_A = make_dataset(x_norm, input_len_A)

# Model B: input_len = 10 (used for autoregressive generation)
input_len_B = 10
X_B, y_B = make_dataset(x_norm, input_len_B)

# train/test split
test_size = 0.2
random_seed = 42
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_A, y_A, test_size=test_size, random_state=random_seed)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_B, y_B, test_size=test_size, random_state=random_seed)

print("Model A shapes:", Xa_train.shape, ya_train.shape, "Model B shapes:", Xb_train.shape, yb_train.shape)

# ---------- Build models ----------
def build_simple_rnn(input_len, hidden_size=32):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_len,1)),
        tf.keras.layers.SimpleRNN(hidden_size, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',
                  metrics=['mse'])
    return model

model_A = build_simple_rnn(input_len_A, hidden_size=32)
model_B = build_simple_rnn(input_len_B, hidden_size=64)

print("Model A summary:")
model_A.summary()
print("\nModel B summary:")
model_B.summary()

# ---------- Train ----------
epochs_A = 30
epochs_B = 40

hist_A = model_A.fit(Xa_train, ya_train, validation_data=(Xa_test, ya_test),
                     epochs=epochs_A, batch_size=32, verbose=1)

hist_B = model_B.fit(Xb_train, yb_train, validation_data=(Xb_test, yb_test),
                     epochs=epochs_B, batch_size=32, verbose=1)

# ---------- Plot training curves ----------
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.plot(hist_A.history['loss'], label='train')
plt.plot(hist_A.history['val_loss'], label='val')
plt.title('Model A loss')
plt.xlabel('epoch'); plt.ylabel('mse'); plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_B.history['loss'], label='train')
plt.plot(hist_B.history['val_loss'], label='val')
plt.title('Model B loss')
plt.xlabel('epoch'); plt.ylabel('mse'); plt.legend()

plt.tight_layout()
plt.show()

# ---------- Evaluate one-step predictions ----------
preds_A = model_A.predict(Xa_test)
preds_A_un = preds_A.flatten() * std_x + mean_x
ya_test_un = ya_test.flatten() * std_x + mean_x

print("Model A one-step MSE (unnormalized):", np.mean((preds_A_un - ya_test_un)**2))

plt.figure(figsize=(8,3))
nplot = min(100, len(ya_test_un))
plt.plot(ya_test_un[:nplot], label='true next x')
plt.plot(preds_A_un[:nplot], label='predicted next x (Model A)')
plt.title("Model A: one-step predictions (segment)")
plt.legend()
plt.show()

# ---------- Autoregressive generation using Model B ----------
# Start from the first input_len_B true values, then generate the remainder autoregressively
initial_window = x_norm[:input_len_B].reshape(1,input_len_B,1)
gen_steps = len(x_norm) - input_len_B
generated = []
current_window = initial_window.copy()

for i in range(gen_steps):
    pred_norm = model_B.predict(current_window, verbose=0)  # shape (1,1)
    generated.append(pred_norm.flatten()[0])
    # roll the window and append prediction
    current_window = np.concatenate([current_window[:,1:,:], pred_norm.reshape(1,1,1)], axis=1)

generated_un = np.array(generated) * std_x + mean_x
true_remainder = x[input_len_B:]

plt.figure(figsize=(8,3))
plt.plot(true_remainder, label='true remainder')
plt.plot(generated_un, label='generated (Model B)')
plt.title('Model B autoregressive generation')
plt.legend()
plt.show()

# ---------- Save models ----------
os.makedirs('saved_models', exist_ok=True)
path_A = os.path.join('saved_models','model_A_rnn.h5')
path_B = os.path.join('saved_models','model_B_rnn.h5')
model_A.save(path_A)
model_B.save(path_B)
print("Saved models to:", path_A, path_B)

# ---------- Final numeric summaries ----------
preds_B = model_B.predict(Xb_test)
preds_B_un = preds_B.flatten() * std_x + mean_x
yb_test_un = yb_test.flatten() * std_x + mean_x
mse_A = np.mean((preds_A_un - ya_test_un)**2)
mse_B = np.mean((preds_B_un - yb_test_un)**2)
print(f"One-step MSE (Model A): {mse_A:.6e}")
print(f"One-step MSE (Model B): {mse_B:.6e}")
