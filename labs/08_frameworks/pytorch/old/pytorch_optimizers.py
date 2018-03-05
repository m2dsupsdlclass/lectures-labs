from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import mse_loss
from torch.optim import Adam


class Quadratic(nn.Module):
    def __init__(self, q: torch.FloatTensor, l: torch.FloatTensor,
                 c: torch.FloatTensor) -> None:
        # Always do this to use PyTorch autograd logic
        super().__init__()

        assert (q.shape == (2, 2))
        assert (l.shape == (2,))
        assert (c.shape == (1,))

        self.q = Parameter(q, requires_grad=False)
        self.l = Parameter(l, requires_grad=False)
        self.c = Parameter(c, requires_grad=False)

    def forward(self, x: Variable) -> Variable:
        assert (x.ndimension() == 2)  # First dimension is the batch size
        assert (x.shape[1] == 2)
        value = (torch.sum((x @ self.q) * x, dim=1)
                 + torch.sum(x * self.l, dim=1) + self.c)
        return value


class Gaussian(nn.Module):
    def __init__(self, precision: torch.FloatTensor, mean: torch.FloatTensor):
        super().__init__()

        assert (precision.shape == (2, 2))
        assert (mean.shape == (2,))

        self.precision = Parameter(precision, requires_grad=False)
        self.mean = Parameter(mean, requires_grad=False)

    def forward(self, x: Variable):
        """Compute the likelihood of x given a Gaussian model.

        https://en.wikipedia.org/wiki/Multivariate_normal_distributions

        """
        # determinant = torch.potrf(self.precision).diag().prod()
        # scale = (2 * math.pi) / torch.sqrt(determinant)
        xc = x - self.mean
        value = torch.exp(- .5 * (torch.sum((xc @ self.precision)
                                            * xc, dim=1)))
        return value  # / scale


class GaussianCombination(nn.Module):
    def __init__(self, precisions: List[torch.FloatTensor],
                 means: List[torch.FloatTensor],
                 weights: List[float]) -> None:
        super().__init__()
        assert (len(precisions) == len(means) == len(weights))
        self.gaussians = nn.ModuleList()
        for precision, mean in zip(precisions, means):
            self.gaussians.append(Gaussian(precision, mean))
        self.weights = weights

    def forward(self, x: Variable) -> Variable:
        res = 0
        for gaussian, weight in zip(self.gaussians, self.weights):
            res += gaussian(x) * weight
        return res


class GradientDescent:
    def __init__(self, params, lr=object(),
                 ):
        self.params = params
        self.lr = lr
        self.momentum = 0

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()


class MomentumGradientDescent(GradientDescent):
    def __init__(self, params, lr, momentum=.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [param.data.new(param.shape).zero_()
                           for param in params]

    def step(self):
        # TODO
        for i, (param, velocity) in enumerate(zip(self.params,
                                                  self.velocities)):
            velocity = self.momentum * velocity + (1 - self.momentum) \
                       * param.grad.data
            param.data = param.data - self.lr * velocity
            self.velocities[i] = velocity


def minimize(f: nn.Module, lr: float, init: torch.FloatTensor,
             optimizer: str = 'gd', max_iter: int = 100) \
        -> Tuple[torch.FloatTensor, torch.FloatTensor,]:
    # Let us now use the power of autograd to write a very
    # simple gradient descent algorithm
    # Initialization: iterate is a Tensor
    iterate = init[None, :]
    iterate = Variable(iterate, requires_grad=True)
    if optimizer == 'gd':
        optimizer = GradientDescent([iterate], lr=lr)
    elif optimizer == 'momentum':
        optimizer = MomentumGradientDescent([iterate], lr=lr)
    elif optimizer == 'adam':
        optimizer = Adam([iterate], lr=lr)
    else:
        raise ValueError('Wrong parameter for option `optimizer`, got %s.'
                         % optimizer)

    value_rec = []
    iterate_rec = []

    for i in range(max_iter):
        optimizer.zero_grad()
        # We wrap x in a Variable, of which we will compute the gradient
        # We compute the value of the quadratic
        value = f(iterate)
        # Compute the gradient of value (a scalar) w.r.t. all
        #  Variable/Parameter for which requires_grad=True
        # x has two main attributes: x.grad = \pdiff{value}{x} (a Variable),
        # x.data
        value.backward()
        value_rec.append(value.data)
        iterate_rec.append(iterate.data.clone())
        print('Iteration %i: f(x) = %e, x = [%e, %e]'
              % (i, value, iterate[0, 0], iterate[0, 1]))
        optimizer.step()
    value_rec = torch.cat(value_rec, dim=0)
    iterate_rec = torch.cat(iterate_rec, dim=0)
    return value_rec, iterate_rec


def plot_function(f, ax):
    x_max, y_max, x_min, y_min = 3, 3, -3, -3
    x = np.linspace(x_min, x_max, 100, dtype=np.float32)
    y = np.linspace(y_min, y_max, 100, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    samples = np.concatenate((X[:, :, None], Y[:, :, None]), axis=2)
    samples = samples.reshape(-1, 2)
    samples = Variable(torch.from_numpy(samples), requires_grad=False)
    Z = f(samples).data.numpy()
    Z = Z.reshape(100, 100)
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)


def plot_trace(iterate_rec, ax, label=None):
    iterate_rec = iterate_rec.numpy()
    line = ax.plot(iterate_rec[:, 0], iterate_rec[:, 1], linestyle=':',
            marker='o', markersize=2, label=label)
    color = plt.getp(line[0], 'color')
    bbox_props = dict(boxstyle="square,pad=0.3", ec=color, fc='white',
                      lw=1)
    for i in range(0, len(iterate_rec), len(iterate_rec) // 10):
        ax.annotate(i, xy=(iterate_rec[i, 0], iterate_rec[i, 1]),
                    xycoords='data',
                    xytext=(5 + np.random.uniform(-2, 2),
                            5 + np.random.uniform(-2, 2)),
                    textcoords='offset points',
                    bbox=bbox_props
                    )


def minimize_gaussian_diff():
    p1 = torch.FloatTensor([[1, 0], [0, 4]])
    m1 = torch.FloatTensor([0, 1])
    w1 = 1
    p2 = torch.FloatTensor([[2, 0], [0, 20]])
    m2 = torch.FloatTensor([0, -1])
    w2 = - 1

    f = GaussianCombination([p1, p2], [m1, m2], [w1, w2])

    # init = torch.FloatTensor([1, .9])
    init = torch.FloatTensor([.8, .8])
    # init = torch.FloatTensor([1, -1.2])
    lr = .1

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_function(f, ax)
    value_rec, iterate_rec = minimize(f, lr, init=init, max_iter=200,
                                      optimizer='gd')
    plot_trace(iterate_rec, ax, label='gd')
    value_rec, iterate_rec = minimize(f, lr, init=init, max_iter=200,
                                      optimizer='momentum')
    plot_trace(iterate_rec, ax, label='momentum')
    value_rec, iterate_rec = minimize(f, lr, init=init, max_iter=100,
                                      optimizer='adam')
    plot_trace(iterate_rec, ax, label='adam')
    plt.legend()
    plt.show()


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.w1 = Parameter(torch.FloatTensor((1, )))
        self.w2 = Parameter(torch.FloatTensor((1, )))

    def reset_parameters(self):
        self.w1.data.uniform(-.1, .1)
        self.w2.data.uniform(-.1, .1)

    def forward(self, x):
        return self.w1 * self.w2 * self.x


def generate_data(n_samples=100):
    x = torch.FloatTensor((n_samples,)).uniform_(-1, 1)
    epsilon = torch.FloatTensor((n_samples,)).normal_(0, 1)
    y = 2 * x + epsilon
    return x, y


def train():
    x, y = generate_data()
    model = SimpleMLP()
    pred = model(x)
    loss = mse_loss(pred, y, reduce=False)



if __name__ == '__main__':
    minimize_gaussian_diff()
