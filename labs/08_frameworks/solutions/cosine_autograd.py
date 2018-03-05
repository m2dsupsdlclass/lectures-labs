def g(x, y):
    xy = torch.dot(x, y)
    norms = f(x) * f(y)
    return xy / norms


x = Variable(torch.FloatTensor([0, 1, 2]), requires_grad=True)
y = Variable(torch.FloatTensor([3, 0.9, 2.2]), requires_grad=True)

cosine = g(x, y)
cosine.backward()
x.grad, y.grad