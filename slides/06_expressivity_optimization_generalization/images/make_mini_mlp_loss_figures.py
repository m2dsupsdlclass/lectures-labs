import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Generate the figures in the same folder
os.chdir(os.path.dirname(__file__))

rng = np.random.RandomState(42)

# 2D parameter space:
n_steps = 200
w1 = np.linspace(-2.5, 2.5, n_steps)
w2 = np.linspace(-2.5, 2.5, n_steps)
w1, w2 = np.meshgrid(w1, w2)


def relu(x):
    return np.maximum(x, 0)


def mini_mlp(x, w1, w2):
    return w2 * relu(w1 * x)


# 30 points dataset
n = 30
x = np.abs(rng.randn(n) + 1)
y = 2 * x + 0.5 * rng.randn(n)  # f(x) = 2x + noise

loss = 0.
for x_i, y_i in zip(x, y):
    loss += (y_i - mini_mlp(w1, w2, x_i)) ** 2
loss /= len(x)

# Plot output surface
fig = plt.figure(figsize=(8, 8))
plt.pcolormesh(w1, w2, loss, cmap=plt.cm.afmhot_r)
plt.contour(w1, w2, loss, 40, colors='k', alpha=0.3)
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Loss function of a ReLU net with 2 params')
fig.savefig("full_data_mlp_loss_landscape.png", dpi=80)


# SGD loss
vmin = 0
vmax = loss.max() * 1.5
folder = "tmp_loss_frames"
shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder)

for i in range(len(x)):
    loss_i = (y[i] - mini_mlp(w1, w2, x[i])) ** 2
    fig = plt.figure(figsize=(8, 8))
    cmesh = plt.pcolormesh(w1, w2, loss_i, vmin=vmin, vmax=vmax,
                           cmap=plt.cm.afmhot_r)
    contour = plt.contour(w1, w2, loss_i, 40, colors='k', alpha=0.3)
    plt.text(-2, 1, "x = %0.2f ; y = %0.2f" % (x[i], y[i]))
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.title('Loss function of a ReLU net with 2 params')
    filename = '%s/loss_%03d.png' % (folder, i)
    print('saving %s...' % filename)
    fig.savefig(filename, dpi=80)


cmd = ("convert -resize 640x640 -delay 100 -loop 0 %s/*.png"
       " sgd_mlp_loss_landscape.gif" % folder)
print(cmd)
os.system(cmd)
shutil.rmtree(folder, ignore_errors=True)
