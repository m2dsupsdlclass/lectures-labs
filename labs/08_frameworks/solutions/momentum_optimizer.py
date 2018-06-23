class MomentumGradientDescent(GradientDescent):

    def __init__(self, params, lr=0.1, momentum=.9):
        super(MomentumGradientDescent, self).__init__(params, lr)
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param, requires_grad=False)
                           for param in params]

    def step(self):
        with torch.no_grad():
            for i, (param, velocity) in enumerate(zip(self.params,
                                                      self.velocities)):
                velocity = self.momentum * velocity + param.grad
                param -= self.lr * velocity
                self.velocities[i] = velocity
