import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

n_hidden = 50
n_layers = 3

m = Sequential()
m.add(Dense(n_hidden, activation='relu', input_dim=2,
            weights=[np.random.randn(2, n_hidden),
                     np.random.randn(n_hidden) / 2]))
for l in range(n_layers - 1):
    m.add(Dense(n_hidden, activation='relu',
                weights=[np.random.randn(n_hidden, n_hidden),
                         np.random.randn(n_hidden) / 2]))
m.add(Dense(1, weights=[np.random.randn(n_hidden, 1),
                        np.random.randn(1) / 2]))

fig = plt.figure(figsize=(12, 10))

# Make data.
x1 = np.linspace(-.5, .5, 100)
x2 = np.linspace(-.5, .5, 100)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T
Y = m.predict(X, batch_size=256).reshape(100, 100)
plt.pcolormesh(X1, X2, Y)
# plt.imshow(Y)
