import torch
import slangpy

m = slangpy.loadModule("linear.slang")

x = torch.Tensor([[3., 0.], [0., 0.], [0., 0.]])
x = x.to(device='cuda:0')
print(f"X = {x}")
y = torch.Tensor([1., 2., 5.])
y = y.to(device='cuda:0')
print(f"Y = {y}")
z = m.linear_fwd(y, x)
print(f"Z = {z}")