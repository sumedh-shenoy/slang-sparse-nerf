import torch
import slangpy

m = slangpy.loadModule("linear.slang")

class SlangLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return m.linear_fwd(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        [input, weight] = ctx.saved_tensors
        input_grad = m.linear_bwd_input(input, weight, grad_output)
        weight_grad = m.linear_bwd_weight(input, weight, grad_output)
        return input_grad, weight_grad

w = torch.tensor([[100000.0, 10000.0],[1000.0, 100.0], [10.0, 1.0]], requires_grad=True, device='cuda:0')
print(f"W = {w}")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device='cuda:0')
print(f"X = {x}")
y_pred = SlangLinearLayer.apply(x, w)
loss = y_pred.sum()
loss.backward()
print(f"Y = {y_pred}")
print(f"dX = {x.grad.cpu()}")
print(f"dW = {w.grad.cpu()}")