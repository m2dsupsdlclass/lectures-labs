x = torch.tensor([0, 1, 2], dtype=torch.float32, requires_grad=True)
y = torch.tensor(2 * x.data.clone(), requires_grad=True)

cosine = g(x, y)
cosine.backward()
x.grad, y.grad