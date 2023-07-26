import torch
import math
import slangpy

m = slangpy.loadModule("new_mlp.slang")


"""
w1 = torch.Tensor([[3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]).to(device='cuda:0') # 3 x 10
w2 = torch.Tensor([[1.0, 0.5, 10.], [0.0, 5.0, -3.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.]]).to(device='cuda:0') # 10 x 3
b1 = torch.Tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).to(device='cuda:0')
b2 = torch.Tensor([2., 3., 4.]).to(device='cuda:0')

w1 = w1/30.
w2 = w2/30.
b1 = b1/30.
b2 = b2/30.
"""


w0 = (torch.rand(3, 32) * 2 / math.sqrt(3) - 1 / math.sqrt(3)).to(device='cuda:0')
w1 = (torch.rand(32, 32) * 2 / math.sqrt(32) - 1 / math.sqrt(32)).to(device='cuda:0')
w2 = (torch.rand(32, 3) * 2 / math.sqrt(32) - 1 / math.sqrt(32)).to(device='cuda:0')

b0 = (torch.rand(32,) * 2 / math.sqrt(3) - 1 / math.sqrt(3)).to(device='cuda:0')
b1 = (torch.rand(32,) * 2 / math.sqrt(32) - 1 / math.sqrt(32)).to(device='cuda:0')
b2 = (torch.rand(3,) * 2 / math.sqrt(32) - 1 / math.sqrt(32)).to(device='cuda:0')


x = torch.Tensor([[0.5, 0.11, 0.4], [0.35, -0.2, 0.1], [0.35, 0.2, 0.1]]).to(device='cuda:0')
x2 = torch.Tensor([[0.5, 0.11, 0.4], [0.35, -0.2, 0.1], [0.35, 0.2, 0.1]]).to(device='cuda:0')
x2.requires_grad = True
activations = [1, 1, 1]

model = torch.nn.Sequential(torch.nn.Linear(3, 32, bias=True), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3), torch.nn.ReLU())
with torch.no_grad():
    model[0].weight = torch.nn.Parameter(w0.clone().T)
    model[0].bias = torch.nn.Parameter(b0.clone().T)
    model[2].weight = torch.nn.Parameter(w1.clone().T)
    model[2].bias = torch.nn.Parameter(b1.clone().T)
    model[4].weight = torch.nn.Parameter(w2.clone().T)
    model[4].bias = torch.nn.Parameter(b2.clone().T)
    

result = torch.Tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]).to(device='cuda:0')

loss_fn = torch.nn.MSELoss(reduction='mean')
f = m.mlp_fwd(x, [w0, w1, w2], [b0, b1, b2], activations)
output = model(x2)

f.requires_grad = True

loss1 = loss_fn(f, result)
loss1.backward()

loss2 = loss_fn(output, result)
loss2.backward()

z = m.mlp_bwd(x, [w0, w1, w2], [b0, b1, b2], activations, f.grad)

with torch.no_grad():
    print(f - output)
    print(z[0][0] - x2.grad)
    #print(x2.grad)
    #print(z[0][0])
    
    for i in range(3):
        print(z[0][i + 1] - model[2*i].weight.grad.T)
    for i in range(3):
        print(z[0][i + 4] - model[2*i].bias.grad.T)
    
