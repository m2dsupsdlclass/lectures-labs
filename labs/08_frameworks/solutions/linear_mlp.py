class LinearMLP(nn.Module):
    def __init__(self, w=None):
        super(LinearMLP, self).__init__()
        self.w1 = Parameter(torch.FloatTensor((1, )))
        self.w2 = Parameter(torch.FloatTensor((1, )))
        if w is None:
            self.reset_parameters()
        else:
            self.set_parameters(w)

    def reset_parameters(self):
        self.w1.data.uniform_(-.1, .1)
        self.w2.data.uniform_(-.1, .1)

    def set_parameters(self, w):
        self.w1.data[0] = w[0]
        self.w2.data[0] = w[1]

    def forward(self, x):
        return self.w1 * self.w2 * x


def expected_risk_linear_mlp(w1, w2):
    return .5 * (8 / 3 - (8 / 3) * w1 * w2 + 2 / 3 * w1 ** 2 * w2 ** 2) + std ** 2


W1, W2, risks, emp_risk, exp_risk = make_grids(
    x, y, LinearMLP, expected_risk_func=expected_risk_linear_mlp)
init = torch.FloatTensor([2, -1.9])
model = LinearMLP(init)
iterate_rec, grad_rec = train(model, x, y, lr=.05, n_epochs=10)
n_iter = len(iterate_rec)
for iter_ in list(range(5)) + list(range(5, n_iter, 40)):
    sample = iter_ % n_samples
    plot_map(W1, W2, risks, emp_risk, exp_risk, sample, iter_)