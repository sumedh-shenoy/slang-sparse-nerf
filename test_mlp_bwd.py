import torch
import slangpy

m = slangpy.loadModule("new_mlp.slang")

w1 = torch.Tensor([[3., 4., 5., 6., 7., 8.], [3., 4., 5., 6., 7., 8.], [3., 4., 5., 6., 7., 8.]]).to(device='cuda:0') # 3 x 6
w2 = torch.Tensor([[1.0, 0.5, 10.], [0.0, 5.0, -3.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.]]) # 6 x 3
b1 = torch.Tensor([1., 2., 3., 4., 5., 6.])
b2 = torch.Tensor([2., 3., 4.])

w2 = w2.to(device='cuda:0')
b1 = b1.to(device='cuda:0')
b2 = b2.to(device='cuda:0')

x = torch.Tensor([[-1., -2., -3.]])
x = x.to(device='cuda:0')
activations = [0, 0]

result = torch.Tensor([[1., 1., 1.]]).to(device='cuda:0')

f = m.mlp_fwd(x, [w1, w2], [b1, b2], activations)
print(f)
z = m.mlp_bwd(x, [w1, w2], [b1, b2], activations, result)
# print(z)


