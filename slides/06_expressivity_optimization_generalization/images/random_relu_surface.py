import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

n_hidden = 50
n_layers = 3
s = 0.01


def weight_init(n_in, n_out):
    return s * np.random.randn(n_in, n_out)


def bias_init(n):
    return s * np.random.randn(n) / 100
    # return np.zeros(n)


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

fig = plt.figure(figsize=(12, 10))

# Make data.
x1 = np.linspace(-.5, .5, 100)
x2 = np.linspace(-.5, .5, 100)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T
Y = m.predict(X, batch_size=256).reshape(100, 100)
plt.pcolormesh(X1, X2, Y)
plt.contour(X1, X2, Y, 6, colors='k')

plt.figure(figsize=(12, 3))
plt.hist(Y.ravel(), bins=50)
print(Y.min(), Y.max())
# plt.imshow(Y)
