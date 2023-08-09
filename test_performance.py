import torch
import slangpy
import numpy as np
import math
import os
import sys
import time

from common import read_image, write_image

image_name = "test_6x8.png"
image_file = "data/images/" + image_name
result_filename = "data/outputs/" + image_name
reconstructed_file2 = "data/outputs/t_" + image_name

mlp_num_layers = 3
output_size = 3
n_levels = 16
n_features = 2
per_level_scale = 2.0
hashmap_size = 1 << 17
base_resolution = 16

full_fused = slangpy.loadModule("fully_fused_mlp_2d.slang")
mlp_construction = slangpy.loadModule("mlp.slang")

class mlp_hash_encoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation_layers, input, encoding, *weights):
        ctx.save_for_backward(input, encoding, *weights)

        weight = weights[:mlp_num_layers]
        bias = weights[mlp_num_layers:]
        ctx.activations = activation_layers

        value = full_fused.mlp_fwd(input, encoding, weight, bias, activation_layers)
        return value

    @staticmethod
    def backward(ctx, grad_out):
        parameters = ctx.saved_tensors
        activations = ctx.activations

        input = parameters[0]
        encoding = parameters[1]
        weight = parameters[2:2 + mlp_num_layers]
        bias = parameters[2 + mlp_num_layers:]

        gradients = full_fused.mlp_bwd(input, encoding, weight, bias, activations, grad_out)
        return None, *gradients[0]

class fully_fused_mlp(torch.nn.Module):
    def __init__(self, n_neurons, activations):
        super(fully_fused_mlp, self).__init__()
        self.fn = mlp_hash_encoding.apply

        if n_neurons[0] != n_levels * n_features or n_neurons[mlp_num_layers] != output_size:
            raise ValueError

        self.activations = activations
        encoding = torch.nn.Parameter((torch.rand(hashmap_size, n_levels, n_features) * 2 / 10**4 - 1/ 10**4).to(device='cuda:0'))
        weights =  [torch.nn.Parameter((torch.rand(n_neurons[i], n_neurons[i+1]) * 2 / math.sqrt(n_neurons[i]) - 1 / math.sqrt(n_neurons[i])).to(device='cuda:0')) for i in range(mlp_num_layers)]
        biases = [torch.nn.Parameter((torch.rand(n_neurons[i+1],) * 2 / math.sqrt(n_neurons[i]) - 1 / math.sqrt(n_neurons[i])).to(device='cuda:0')) for i in range(mlp_num_layers)]

        self.params = torch.nn.ParameterList([encoding, *weights, *biases])

    def forward(self, x):
        return self.fn(self.activations, x, self.params[0], *self.params[1:])

class custom_mlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation_layers, input, *weights):
        ctx.save_for_backward(input, *weights)

        weight = weights[:mlp_num_layers]
        bias = weights[ mlp_num_layers:2*mlp_num_layers]
        ctx.activations = activation_layers

        value = mlp_construction.mlp_fwd(input, weight, bias, activation_layers)

        return value

    @staticmethod
    def backward(ctx, grad_out):
        vals = ctx.saved_tensors
        activations = ctx.activations

        input = vals[0]
        weights = vals[1:(1 + mlp_num_layers)]
        bias = vals[(1 + mlp_num_layers):(1 + 2*mlp_num_layers)]

        ret = mlp_construction.mlp_bwd(input, weights, bias, activations, grad_out)
        return None, *ret[0]

class mlp(torch.nn.Module):
    def __init__(self, size_ins, activations):
        super(mlp, self).__init__()
        self.fn = custom_mlp.apply
        size_ins.append(output_size)

        self.activations = activations
        self.weights = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(size_ins[i], size_ins[i+1]) * 2 / math.sqrt(size_ins[i]) - 1 / math.sqrt(size_ins[i])).to(device='cuda:0')) for i in range(mlp_num_layers)])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(size_ins[i+1],) * 2 / math.sqrt(size_ins[i]) - 1 / math.sqrt(size_ins[i])).to(device='cuda:0')) for i in range(mlp_num_layers)])
        self.all_weights = torch.nn.ParameterList([*self.weights, *self.biases])

    def forward(self, x):
        return self.fn(self.activations, x, *self.all_weights)

class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()
			lerp_weights = xs - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)



if __name__=="__main__":
    device = 'cuda:0'
    print(hashmap_size)
    image = Image(image_file, device)
    print(image.shape)

    n_channels = image.shape[2]
    # model = fully_fused_mlp([32, 64, 64, 3], [1, 1, 1])
    model = torch.nn.Sequential(torch.nn.Linear(2, 64).to(device='cuda:0'), torch.nn.ReLU(), torch.nn.Linear(64, 64).to(device='cuda:0'), torch.nn.ReLU(), torch.nn.Linear(64, 3).to(device='cuda:0'), torch.nn.ReLU())
    # model = mlp([2, 64, 64], [1, 1, 1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Variables for saving/displaying image results
    resolution = image.data.shape[0:2]
    img_shape = resolution + torch.Size([image.data.shape[2]])
    n_pixels = resolution[0] * resolution[1]

    half_dx =  0.5 / resolution[0]
    half_dy =  0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    path = f"data/reference.jpg"
    print(f"Writing '{path}'... ", end="")
    model(xy)
    write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    print("done.")

    prev_time = time.perf_counter()

    batch_size = 2**18
    interval = 50
    interval_mid = 45

    n_steps = interval + 1

    print(f"Beginning optimization with {n_steps} training steps.")

    try:
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        traced_image = torch.jit.trace(image, batch)
    except:
        # If tracing causes an error, fall back to regular execution
        print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
        traced_image = image

    for i in range(n_steps):
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        targets = traced_image(batch)
        output = model(batch)

        # relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
        # loss = relative_l2_error.mean()
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """

        if i % interval == 0 or i == interval_mid:
            """
            loss_val = loss.item()
            torch.cuda.synchronize()
            """
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={0} time={int(elapsed_time*1000000)}[µs]")
            prev_time = time.perf_counter()

            """
            path = f"data/{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            
            with torch.no_grad():
                write_image(path, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()
            """

            """
            if i > 0 and interval < 1000:
                interval *= 10
            """

    if result_filename:
        print(f"Writing '{result_filename}'... ", end="")
        with torch.no_grad():
            write_image(result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
        print("done.")
