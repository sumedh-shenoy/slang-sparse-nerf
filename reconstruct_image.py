import torch
import slangpy
import numpy as np
import math
import os
import sys
import time

from torchvision.io import read_image, write_png, image

image_name = "test_6x8.png"
image_file = "data/images/" + image_name
reconstructed_file = "/data/outputs/" + image_name

mlp_num_layers = 3
output_size = 3
num_features = 4
rescale_factor = 4

mlp_construction = slangpy.loadModule("mlp.slang", defines={"max_size": 32, "output_size": output_size, "num_layers": mlp_num_layers})
encoding = slangpy.loadModule("feature_encoding.slang")

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
        return None, ret[0], *ret[1], *ret[2]

class sample_feature_grid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature_grid, location):
        ctx.save_for_backward(feature_grid, location)
        return encoding.feature_encoding_fwd(feature_grid, location)

    @staticmethod
    def backward(ctx, grad_out):
        [feature_grid, location] = ctx.saved_tensors
        feature_gradient = encoding.feature_encoding_bwd(feature_grid, location, grad_out)
        return feature_gradient, None

"""
Simple image sampling module, taken from tiny-cuda-nn sample code. 
"""
class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename, mode=image.ImageReadMode.RGB).to(device=device)
		self.data = torch.transpose(self.data, 0, 2)
		self.shape = self.data.size()

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

class mlp(torch.nn.Module):
    def __init__(self, size_ins, activations):
        super(mlp, self).__init__()
        self.fn = custom_mlp.apply
        size_ins.append(output_size)

        self.activations = activations
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(size_ins[i], size_ins[i+1]).to(device='cuda:0')) for i in range(mlp_num_layers)])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(size_ins[i+1],).to(device='cuda:0')) for i in range(mlp_num_layers)])
        self.all_weights = torch.nn.ParameterList([*self.weights, *self.biases])

    def forward(self, x):
        return self.fn(self.activations, x, *self.all_weights)

class feature_encoding(torch.nn.Module):
    def __init__(self, image_size):
        super(feature_encoding, self).__init__()
        image_size[0] = math.ceil(image_size[0]/rescale_factor)
        image_size[1] = math.ceil(image_size[1]/rescale_factor)

        self.fn = sample_feature_grid.apply
        self.feature_grid = torch.nn.Parameter(torch.randn(image_size[0], image_size[1], num_features).to(device='cuda:0'))
    
    def forward(self, location):
        return self.fn(self.feature_grid, location)



if __name__ == "__main__":
    device = 'cuda:0'
    img = Image(image_file, device)

    # Variables for saving results

    
    im_width = img.shape[0]
    im_height = img.shape[1]
    
    im_shape = [3, im_height, im_width]

    xs = torch.linspace(0.5, im_width-0.5, im_width)
    ys = torch.linspace(0.5, im_height-0.5, im_height)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((xv.flatten(), yv.flatten())).t().to(device=device)

    learning_rate = 0.01
    
    model = torch.nn.Sequential(feature_encoding([im_width, im_height]), mlp([4, 32, 32], [1, 1, 1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_steps = 5
    batch_size = 1
    batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
    print(img(batch))
    

    prev_time = time.perf_counter()
    
    for i in range(num_steps):
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        targets = img(batch)
        output = model(batch)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, targets)
        print(loss)
        print("we output", output)
        print("we seek", targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        elapsed_time = time.perf_counter() - prev_time
        print("Time taken:", elapsed_time)
        output = model(xy).reshape(im_shape).clamp(0., 255.).detach()
        print(output)
        write_png(output.cpu(), reconstructed_file)
    
    
    """
    model = torch.nn.Sequential(feature_encoding([8, 8]), mlp([4, 4, 4, 4, 4], [1, 1, 1, 1, 1]))
    print(list(model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(1000):
        X = torch.Tensor([[3., 3.], [1., 1.]]).to(device='cuda:0')
        Y = torch.Tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]).to(device='cuda:0')
        pred = model(X)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, Y)
        if i%250 == 0:
            print(pred)
            print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        print(list(model.parameters()))
    """



    


        


