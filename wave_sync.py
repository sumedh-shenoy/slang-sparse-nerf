import torch
import math
import slangpy

m = slangpy.loadModule("mlp.slang")

# for reproducability 
torch.manual_seed(0)

batch_size = 64

w0 = (torch.rand(2, 64) * 2 / math.sqrt(2) - 1 / math.sqrt(2)).to(device='cuda:0')
w1 = (torch.rand(64, 64) * 2 / math.sqrt(64) - 1 / math.sqrt(64)).to(device='cuda:0')
w2 = (torch.rand(64, 3) * 2 / math.sqrt(64) - 1 / math.sqrt(64)).to(device='cuda:0')

b0 = (torch.rand(64,) * 2 / math.sqrt(3) - 1 / math.sqrt(3)).to(device='cuda:0')
b1 = (torch.rand(64,) * 2 / math.sqrt(64) - 1 / math.sqrt(64)).to(device='cuda:0')
b2 = (torch.rand(3,) * 2 / math.sqrt(64) - 1 / math.sqrt(64)).to(device='cuda:0')


x = torch.rand(batch_size, 2).to(device='cuda:0')

x2 = x.clone().detach().to(device='cuda:0')
x2.requires_grad = True
activations = [1, 1, 1]

model = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3), torch.nn.ReLU())
with torch.no_grad():
    model[0].weight = torch.nn.Parameter(w0.clone().T)
    model[0].bias = torch.nn.Parameter(b0.clone().T)
    model[2].weight = torch.nn.Parameter(w1.clone().T)
    model[2].bias = torch.nn.Parameter(b1.clone().T)
    model[4].weight = torch.nn.Parameter(w2.clone().T)
    model[4].bias = torch.nn.Parameter(b2.clone().T)


result = (torch.ones(batch_size, 3) / 2.0).to(device='cuda:0')

loss_fn = torch.nn.MSELoss(reduction='mean')
f = m.mlp_fwd(x, [w0, w1, w2], [b0, b1, b2], activations)[:batch_size, :]
output = model(x2)

f.requires_grad = True
output.retain_grad()

loss1 = loss_fn(f, result)
loss1.backward()

loss2 = loss_fn(output, result)
loss2.backward()

z = m.mlp_bwd(x, [w0, w1, w2], [b0, b1, b2], activations, f.grad)



with torch.no_grad():
    print(torch.max(f - output))
    print(torch.max(z[0][0][:batch_size, :] - x2.grad))

    print(z[0][6])
