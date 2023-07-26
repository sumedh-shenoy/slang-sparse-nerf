#define max_size 64
#define output_size 3
#define num_layers 3

/*
Other issues: changing the max_size seems to create issues——gradients are calculated correctly (in the test_mlp_bwd.py file, measured against gradients calculated 
by making an MLP w/ linear layers). For 18 to 3something (at least 32), it works, but starts incorrectly calculating gradients when set to 16 or 64. 
*/

struct float_arr : IDifferentiable {
    float vals[output_size];
}

struct mlp_gradients {
    TorchTensor<float> gradients[2*num_layers + 1];
}

struct result_arr : IDifferentiable {
    float vals[max_size];
}

[BackwardDifferentiable]
result_arr mlp_forward(TensorView<float> input, TensorView<float> weight [num_layers], TensorView<float> bias [num_layers], TensorView<float> input_gradient, 
    TensorView<float> weight_gradient [num_layers], TensorView<float> bias_gradient [num_layers], no_diff int activation [num_layers], no_diff uint batch) {
    
    result_arr output = linear_layer_tensor(input, weight[0], bias[0], input_gradient, weight_gradient[0], bias_gradient[0], activation[0], batch);
    
    /*
    Replacing the result_arr temp1 = .... code w/ this for loop results in gradients not being properly calculated, and returns seemingly random values. Multiple runs w/
    the same input test cases result in different gradients, many of which have huge magnitudes (on the order of 10^37...), and others which have the correct gradients. To replicate: set num_layers to 2, and run test_mlp_bwd.py w/ 
    weights of: 

    w1 = torch.Tensor([[3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]).to(device='cuda:0') # 3 x 10
    w2 = torch.Tensor([[1.0, 0.5, 10.], [0.0, 5.0, -3.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.], [1.0, 0.5, 10.]]).to(device='cuda:0') # 10 x 3
    b1 = torch.Tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).to(device='cuda:0')
    b2 = torch.Tensor([2., 3., 4.]).to(device='cuda:0')

    w1 = w1/30.
    w2 = w2/30.
    b1 = b1/30.
    b2 = b2/30.

    And input of
    x = torch.Tensor([[0.5, 0.11, 0.4], [0.35, -0.2, 0.1], [0.35, 0.2, 0.1]]).to(device='cuda:0')
    is used.
    */

    /*
    [MaxIters(num_layers - 1)]
    for(int i = 1; i < num_layers; i++) {
        output = linear_layer_norm(output, weight[i], bias[i], weight_gradient[i], bias_gradient[i], activation[i]);
    }
    */
    
    /*
    Temporary code until issues w/ looping are resolved. Remember to check that this matches the number of layers!
    */
    result_arr temp1 = linear_layer_norm(output, weight[1], bias[1], weight_gradient[1], bias_gradient[1], activation[1]);
    result_arr temp2 = linear_layer_norm(temp1,  weight[2], bias[2], weight_gradient[2], bias_gradient[2], activation[2]);
    
    return temp2;
}

result_arr linear_layer_tensor(TensorView<float> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> input_gradient, 
    TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff int activation, no_diff uint batch) {
    result_arr output;
    uint in_size = weight.size(0);
    uint out_size = weight.size(1);

    for (int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for (int j = 0; j < in_size; j++) {
            sum_value = sum_value + get_input_val(input, input_gradient, uint2(batch, j)) * get_weight_val(weight, weight_gradient, uint2(j, i));
        }
        sum_value = sum_value + get_bias_val(bias, bias_gradient, i);
        output.vals[i] = compute_activation(sum_value, activation);
    }

    return output;
}

[BackwardDerivativeOf(linear_layer_tensor)]
void linear_layer_tensor_bwd(TensorView<float> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> input_gradient, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff int activation, no_diff uint batch, result_arr.Differential output_gradient) {
    uint in_size = weight.size(0);
    uint out_size = weight.size(1);

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for(int j = 0; j < in_size; j++) {
            sum_value = sum_value + get_input_val(input, input_gradient, uint2(batch, j)) * get_weight_val(weight, weight_gradient, uint2(j, i));
        }
        sum_value = sum_value + get_bias_val(bias, bias_gradient, i);
        DifferentialPair<float> preactivation_value = diffPair(sum_value);
        __bwd_diff(compute_activation)(preactivation_value, activation, output_gradient.vals[i]);
        for(int j = 0; j < in_size; j++) {
            get_input_bwd(input, input_gradient, uint2(batch, j), get_weight_val(weight, weight_gradient, uint2(j, i)) * preactivation_value.d);
        }
        for(int j = 0; j < in_size; j++) {
            get_weight_bwd(weight, weight_gradient, uint2(j, i), get_input_val(input, input_gradient, uint2(batch, j)) * preactivation_value.d);
        }
        get_bias_bwd(bias, bias_gradient, i, preactivation_value.d);
    }
}

result_arr linear_layer_norm(result_arr input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation) {
    result_arr output;
    uint in_size = weight.size(0);
    uint out_size = weight.size(1);

    for (int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for (int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.vals[j] * get_weight_val(weight, weight_gradient, uint2(j, i));
        }
        sum_value = sum_value + get_bias_val(bias, bias_gradient, i);
        output.vals[i] = compute_activation(sum_value, activation);
    }

    return output;
}

[BackwardDerivativeOf(linear_layer_norm)]
void linear_layer_norm_bwd(inout DifferentialPair<result_arr> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, result_arr.Differential output_gradient) {
    uint in_size = weight.size(0);
    uint out_size = weight.size(1);
    result_arr.Differential set_derivatives;

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for(int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.p.vals[j] * get_weight_val(weight, weight_gradient, uint2(j, i));
        }
        sum_value = sum_value + get_bias_val(bias, bias_gradient, i);
        DifferentialPair<float> preactivation_value = diffPair(sum_value);
        __bwd_diff(compute_activation)(preactivation_value, activation, output_gradient.vals[i]);
        for(int j = 0; j < in_size; j++) {
            set_derivatives.vals[j] += get_weight_val(weight, weight_gradient, uint2(j, i)) * preactivation_value.d;
        }
        for(int j = 0; j < in_size; j++) {
            get_weight_bwd(weight, weight_gradient, uint2(j, i), input.p.vals[j] * preactivation_value.d);
        }
        get_bias_bwd(bias, bias_gradient, i, preactivation_value.d);
    }
    input = diffPair(input.p, set_derivatives);
}


[BackwardDifferentiable]
float compute_activation(float val, uint activation_type) {
    if (activation_type == 0) {
        // type 0: Identity
        return val;
    } else if (activation_type == 1) {
        // type 1: ReLU
        return max(val, 0);
    }
    return 0;
}

float get_input_val(TensorView<float> input, TensorView<float> input_gradient, uint2 location) {
    return input[location];
}

[BackwardDerivativeOf(get_input_val)]
void get_input_bwd(TensorView<float> input, TensorView<float> input_gradient, uint2 location, float derivative) {
    float old_val;
    input_gradient.InterlockedAdd(location, derivative, old_val);
}

float get_weight_val(TensorView<float> weight, TensorView<float> weight_gradient, uint2 location) {
    return weight[location];
}

[BackwardDerivativeOf(get_weight_val)]
void get_weight_bwd(TensorView<float> weight, TensorView<float> weight_gradient, uint2 location, float derivative) {
    float old_val;
    weight_gradient.InterlockedAdd(location, derivative, old_val);
}

float get_bias_val(TensorView<float> bias, TensorView<float> bias_gradient, uint location) {
    return bias[location];
}

[BackwardDerivativeOf(get_bias_val)]
void get_bias_bwd(TensorView<float> bias, TensorView<float> bias_gradient, uint location, float derivative) {
    float old_val;
    bias_gradient.InterlockedAdd(location, derivative, old_val);
}

[CudaKernel]
void mlp_fwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> output, 
    TensorView<float> input_gradient, TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers]) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    result_arr res = mlp_forward(input, weights, bias, input_gradient, weight_gradients, bias_gradients, activation_type, batch);
    for (int i = 0; i < output_size; i++) {
        output[uint2(batch, i)] = res.vals[i];
    }
}

[CudaKernel]
void mlp_bwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient, 
    TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers], TensorView<float> result_gradients) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    result_arr.Differential res;
    for (int i = 0; i < output_size; i++) {
        res.vals[i] = result_gradients[uint2(batch, i)];
    }
    __bwd_diff(mlp_forward)(input, weights, bias, input_gradient, weight_gradients, bias_gradients, activation_type, batch, res);
}

[TorchEntryPoint]
TorchTensor<float> mlp_fwd(TorchTensor<float> input, TorchTensor<float> weights[num_layers], TorchTensor<float> bias[num_layers], int activation_type[num_layers]) {
    var input_gradient = TorchTensor<float>.zerosLike(input);
    TensorView<float> weight_views[num_layers];
    TensorView<float> bias_views[num_layers];
    TorchTensor<float> weight_gradients[num_layers];
    TorchTensor<float> bias_gradients[num_layers];
    TensorView<float> weight_gradient_views[num_layers];
    TensorView<float> bias_gradient_views[num_layers];

    [MaxIters(num_layers)]
    for (int i = 0; i < num_layers; i++) {
        weight_views[i] = weights[i].getView();
        bias_views[i] = bias[i].getView();
        weight_gradients[i] = TorchTensor<float>.zerosLike(weights[i]);
        bias_gradients[i] = TorchTensor<float>.zerosLike(bias[i]);
        weight_gradient_views[i] = weight_gradients[i].getView();
        bias_gradient_views[i] = bias_gradients[i].getView();
    }

    uint batches = input.size(0);
    var output = TorchTensor<float>.alloc(batches, output_size);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_fwd_kernel, blockCount, groupSize)(input, weight_views, bias_views, output, input_gradient, weight_gradient_views, bias_gradient_views, activation_type);
    return output;
}

[TorchEntryPoint]
mlp_gradients mlp_bwd(TorchTensor<float> input, TorchTensor<float> weights[num_layers], TorchTensor<float> bias[num_layers], int activation_type[num_layers], TorchTensor<float> result_gradients) {
    TorchTensor<float> input_gradient = TorchTensor<float>.zerosLike(input);
    TensorView<float> weight_views[num_layers];
    TensorView<float> bias_views[num_layers];
    TorchTensor<float> weight_gradients[num_layers];
    TorchTensor<float> bias_gradients[num_layers];
    TensorView<float> weight_gradient_views[num_layers];
    TensorView<float> bias_gradient_views[num_layers];
    
    [MaxIters(num_layers)]
    for (int i = 0; i < num_layers; i++) {
        weight_views[i] = weights[i].getView();
        bias_views[i] = bias[i].getView();
        weight_gradients[i] = TorchTensor<float>.zerosLike(weights[i]);
        bias_gradients[i] = TorchTensor<float>.zerosLike(bias[i]);
        weight_gradient_views[i] = weight_gradients[i].getView();
        bias_gradient_views[i] = bias_gradients[i].getView();
    }
    
    uint batches = input.size(0);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(input, weight_views, bias_views, input_gradient, weight_gradient_views, bias_gradient_views, activation_type, result_gradients);
    mlp_gradients res;
    res.gradients[0] = input_gradient;
    for (int i = 0; i < num_layers; i++) {
        res.gradients[i+1] = weight_gradients[i];
        res.gradients[i+1+num_layers] = bias_gradients[i];
    }
    return res;
}
