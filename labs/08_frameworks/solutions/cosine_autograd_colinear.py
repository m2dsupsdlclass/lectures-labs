x = Variable(torch.FloatTensor([0, 1, 2]), requires_grad=True)
y = Variable(2 * x.data.clone(), requires_grad=True)

cosine = g(x, y)
cosine.backward()
x.grad, y.grad