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


def dloss_mini_mlp(x, y, w1, w2):
    x, y = float(x), float(y)
    delta = 2 * (w2 * relu(w1 * x) - y)
    dw1 = delta * w2 * x if w1 * x > 0 else 0
    dw2 = delta * relu(w1 * x)
    return dw1, dw2


# 30 points dataset
n = 30
x = np.abs(rng.randn(n) + 1)
y = 2 * x + 0.5 * rng.randn(n)  # f(x) = 2x + noise

total_loss = 0.
for x_i, y_i in zip(x, y):
    total_loss += (y_i - mini_mlp(w1, w2, x_i)) ** 2
total_loss /= len(x)

# Plot output surface
# fig = plt.figure(figsize=(8, 8))
# plt.pcolormesh(w1, w2, total_loss, cmap=plt.cm.afmhot_r)
# plt.contour(w1, w2, total_loss, 40, colors='k', alpha=0.3)
# plt.xlabel('$w_1$')
# plt.ylabel('$w_2$')
# plt.title('Loss function of a ReLU net with 2 params')
# fig.savefig("full_data_mlp_loss_landscape.png", dpi=80)


# Stochastic loss
vmin = 0
vmax = total_loss.max() * 1.5
stochastic_loss_folder = "tmp_single_sample_loss_frames"
shutil.rmtree(stochastic_loss_folder, ignore_errors=True)
os.makedirs(stochastic_loss_folder)
total_loss_folder = "tmp_full_data_loss_frames"
shutil.rmtree(total_loss_folder, ignore_errors=True)
os.makedirs(total_loss_folder)

w1_i, w2_i = 0.2, 1.7
iterates = [(w1_i, w2_i)]
lr = 0.05

for i in range(len(x)):
    loss_i = (y[i] - mini_mlp(w1, w2, x[i])) ** 2
    dw1_i, dw2_i = dloss_mini_mlp(x[i], y[i], w1_i, w2_i)
    print("gradient: %0.3f, %0.3f" % (dw1_i, dw2_i))

    for kind, folder in [('single_sample', stochastic_loss_folder),
                         ('full_data', total_loss_folder)]:
        fig = plt.figure(figsize=(8, 8))
        if kind == 'single_sample':
            cmesh = plt.pcolormesh(w1, w2, loss_i, vmin=vmin, vmax=vmax,
                                   cmap=plt.cm.afmhot_r)
            contour = plt.contour(w1, w2, loss_i, 40, colors='k', alpha=0.3)
            plt.text(-2, 1, "x = %0.2f ; y = %0.2f" % (x[i], y[i]))
        else:
            cmesh = plt.pcolormesh(w1, w2, total_loss, vmin=vmin, vmax=vmax,
                                   cmap=plt.cm.afmhot_r)
            contour = plt.contour(w1, w2, total_loss, 40, colors='k',
                                  alpha=0.3)
        plt.plot([w[0] for w in iterates], [w[1] for w in iterates], '-o',
                 c='k', alpha=0.3)
        plt.plot([w[0] for w in iterates[-1:]], [w[1] for w in iterates[-1:]],
                 'o', c='k')
        plt.arrow(w1_i, w2_i, -dw1_i / 5, -dw2_i / 5,
                  head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.xlabel('$w_1$')
        plt.ylabel('$w_2$')
        plt.title('SGD Loss function of a ReLU net with 2 params')
        filename = '%s/loss_%03d.png' % (folder, i)
        print('saving %s...' % filename)
        fig.savefig(filename, dpi=80)
        plt.close()

    w1_i, w2_i = w1_i - lr * dw1_i, w2_i - lr * dw2_i
    iterates.append((w1_i, w2_i))
    print("new w: %0.3f, %0.3f" % (w1_i, w2_i))


for kind, folder in [('single_sample', stochastic_loss_folder),
                     ('full_data', total_loss_folder)]:
    cmd = ("convert -resize 640x640 -delay 100 -loop 0 %s/*.png"
           " %s_mlp_loss_landscape.gif" % (folder, kind))
    print(cmd)
    os.system(cmd)
    shutil.rmtree(folder, ignore_errors=True)
