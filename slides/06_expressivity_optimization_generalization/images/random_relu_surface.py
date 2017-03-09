import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

n_hidden = 30
n_layers = 3
s = 0.01


def weight_init(n_in, n_out):
    return s * np.random.randn(n_in, n_out)


def bias_init(n):
    # return s * np.random.randn(n) / 100
    return np.zeros(n)


m = Sequential()
m.add(Dense(n_hidden, activation='relu', input_dim=2,
            weights=[weight_init(2, n_hidden),
                     bias_init(n_hidden)]))
for l in range(n_layers - 1):
    m.add(Dense(n_hidden, activation='relu',
                weights=[weight_init(n_hidden, n_hidden),
                         bias_init(n_hidden)]))
m.add(Dense(1, weights=[weight_init(n_hidden, 1),
                        bias_init(1)]))

# Make data.
n_steps = 200
x1 = np.linspace(-.5, .5, n_steps)
x2 = np.linspace(-.5, .5, n_steps)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T

# Make predictions on the grid
Y = m.predict(X, batch_size=256).reshape(n_steps, n_steps)
w = np.abs(Y).max()

# Plot output surface
fig = plt.figure(figsize=(12, 10))
plt.pcolormesh(X1, X2, Y, vmin=-w, vmax=w, cmap=plt.cm.RdBu_r)
plt.contour(X1, X2, Y, 6, colors='k')

# Histogram of output values
plt.figure(figsize=(12, 3))
plt.hist(Y.ravel(), bins=50)
