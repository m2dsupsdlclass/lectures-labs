x = torch.tensor([0, 1, 2], dtype=torch.float32, requires_grad=True)
y = 2 * x.clone().detach()
y.requires_grad_()

cosine = g(x, y)
print("cosine: ", cosine)

cosine.backward()
print('x.grad:', x.grad)
print('y.grad:', y.grad)

# The gradient of the cosine similarity of colinear vectors is
# null w.r.t. each of the two vector as the cosine function
# is already at its global maximum with a value of 1.0.