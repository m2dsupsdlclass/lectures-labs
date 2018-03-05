class MomentumGradientDescent(GradientDescent):

    def __init__(self, params, lr=0.1, momentum=.9):
        super(MomentumGradientDescent, self).__init__(params, lr)
        self.momentum = momentum
        self.velocities = [param.data.new(param.shape).zero_()
                           for param in params]

    def step(self):
        for i, (param, velocity) in enumerate(zip(self.params,
                                                  self.velocities)):
            velocity = (self.momentum * velocity +
                        (1 - self.momentum) * param.grad.data)
            param.data = param.data - self.lr * velocity
            self.velocities[i] = velocity
