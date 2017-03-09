import matplotlib.pyplot as plt
import numpy as np
import os

# Generate the figures in the same folder
os.chdir(os.path.dirname(__file__))


def relu(x):
    return np.maximum(x, 0)


def rect(x, a=1, b=2, h=2, eps=1e-7):
    return h / eps * (relu(x - a) - relu(x - (a + eps))
                      - relu(x - b) + relu(x - (b + eps)))


plt.figure()
x = np.linspace(-3, 3, 1000)
y = rect(x, 0, 1, 1.3)
plt.plot(x, y)

plt.ylim(-0.1, 1.4)
plt.savefig('one-rectangle.svg')

plt.figure()
x = np.linspace(-3, 3, 1000)
y = rect(x, -1, 0, 0.4) + rect(x, 0, 1, 1.3) + rect(x, 1, 2, 0.8)
plt.plot(x, y)

plt.ylim(-0.1, 1.4)
plt.savefig('three-rectangles.svg')
