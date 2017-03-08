import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def tri(x):
    return relu(relu(2 * x) - relu(4 * x - 2))


x = np.linspace(0., 1., 1000)

plt.figure()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(x))
plt.savefig('triangle_x.svg')

plt.figure()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(tri(x)))
plt.savefig('triangle_triangle_x.svg')

plt.figure()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.plot(x, tri(tri(tri(x))))
plt.savefig('triangle_triangle_triangle_x.svg')
