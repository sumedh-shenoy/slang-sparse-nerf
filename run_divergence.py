import torch
import slangpy

m = slangpy.loadModule("divergence_issue.slang")

b0 = (torch.ones(3,)).to(device='cuda:0')
z = m.mlp_bwd(b0)
print(z[0][0])
