def g(x, y):
    xy = torch.dot(x, y)
    norms = f(x) * f(y)
    return xy / norms


x = torch.tensor([0, 1, 2], dtype=torch.float32, requires_grad=True)
y = torch.tensor([3, 0.9, 2.2], dtype=torch.float32, requires_grad=True)

cosine = g(x, y)
cosine.backward()
x.grad, y.grad