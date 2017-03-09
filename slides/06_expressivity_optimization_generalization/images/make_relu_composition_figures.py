import matplotlib.pyplot as plt
import numpy as np
import os

# Generate the figures in the same folder
os.chdir(os.path.dirname(__file__))


def relu(x):
    return np.maximum(x, 0)


def tri(x):
    return relu(relu(2 * x) - relu(4 * x - 2))


x = np.linspace(-.3, 1.3, 1000)

plt.figure()
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(x))
plt.savefig('triangle_x.svg')

plt.figure()
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(tri(x)))
plt.savefig('triangle_triangle_x.svg')

plt.figure()
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(tri(tri(x))))
plt.savefig('triangle_triangle_triangle_x.svg')

plt.figure()
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(tri(tri(tri(x)))))
plt.savefig('triangle_triangle_triangle_triangle_x.svg')
