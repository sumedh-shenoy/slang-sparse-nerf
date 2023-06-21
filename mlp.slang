struct float_arr : IDifferentiable {
    float vals[output_size];
}

struct mlp_gradients {
    TorchTensor<float> input_gradient;
    TorchTensor<float> weight_gradients[num_layers];
    TorchTensor<float> bias_gradients[num_layers];
}


[BackwardDifferentiable]
float compute_linear_pass(float [max_size] input, TensorView<float> weight, TensorView<float> bias, TensorView<float> input_gradient, TensorView<float> weight_gradient, TensorView<float> bias_gradient, uint2 location) {
    float sum_value = 0.0;
    uint batch = location.x;
    uint pos = location.y;

    uint height = weight.size(0);
    [MaxIters(max_size)]
    for (int i = 0; i < height; i++) {
        sum_value = sum_value + input[i] * get_weight_val(weight, weight_gradient, uint2(i, pos)) + get_bias_val(bias, bias_gradient, pos);
    }

    return sum_value;
}

[BackwardDifferentiable]
float compute_activation(float val, no_diff int activation_type) {
    if (activation_type == 0) {
        // type 0: Identity
        return val;
    } else if (activation_type == 1) {
        // type 1: ReLU
        return max(val, 0);
    } else if (activation_type == 2) {
        // type 2: Sigmoid
    }
    return 0;
}

[BackwardDifferentiable]
float_arr mlp_compute_forward(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient,
    TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], no_diff int activation_type[num_layers], no_diff uint batch) {

    uint input_size = input.size(1);
    float input_layer[max_size];
    float output_layer[max_size];

    [MaxIters(max_size)]
    for (int i = 0; i < input_size; i++) {
        input_layer[i] = get_input_val(input, input_gradient, uint2(batch, i));
    }
    
    [MaxIters(num_layers)]
    for (int i = 0; i < num_layers; i++) {
        uint next_height = weights[i].size(1);
        [MaxIters(max_size)]
        for (int j = 0; j < next_height; j++) {
            output_layer[j] = compute_linear_pass(input_layer, weights[i], bias[i], input_gradient, weight_gradients[i], bias_gradients[i], uint2(batch, j));
            output_layer[j] = compute_activation(output_layer[j], activation_type[i]);
        }
        [MaxIters(max_size)]
        for (int j = 0; j < max_size; j++) {
            input_layer[j] = output_layer[j];
        }
    }
    
    float_arr output_val;

    [MaxIters(output_size)]
    for (int i = 0; i < output_size; i++) {
        output_val.vals[i] = output_layer[i];
    }
    return output_val;
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
    float_arr res = mlp_compute_forward(input, weights, bias, input_gradient, weight_gradients, bias_gradients, activation_type, batch);
    for (int i = 0; i < output_size; i++) {
        output[uint2(batch, i)] = res.vals[i];
    }
}

[CudaKernel]
void mlp_bwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient, 
    TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers], TensorView<float> result_gradients) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    float_arr.Differential res;
    for (int i = 0; i < output_size; i++) {
        res.vals[i] = result_gradients[uint2(batch, i)];
    }
    __bwd_diff(mlp_compute_forward)(input, weights, bias, input_gradient, weight_gradients, bias_gradients, activation_type, batch, res);
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
    let blockCount = uint3(2);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_fwd_kernel, blockCount, groupSize)(input, weight_views, bias_views, output, input_gradient, weight_gradient_views, bias_gradient_views, activation_type);
    return output;
}

[TorchEntryPoint]
mlp_gradients mlp_bwd(TorchTensor<float> input, TorchTensor<float> weights[num_layers], TorchTensor<float> bias[num_layers], int activation_type[num_layers], TorchTensor<float> result_gradients) {
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
    let blockCount = uint3(2);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(input, weight_views, bias_views, input_gradient, weight_gradient_views, bias_gradient_views, activation_type, result_gradients);
    mlp_gradients res;
    res.input_gradient = input_gradient;
    for (int i = 0; i < num_layers; i++) {
        res.weight_gradients[i] = weight_gradients[i];
        res.bias_gradients[i] = bias_gradients[i];
    }
    return res;
}

/*
#define max_size 32
#define output_size 3
#define num_layers 2


struct float_arr : IDifferentiable {
    float vals[output_size];
}

struct tensor_arr {
    TorchTensor<float> vals[num_layers + 1];
}


[BackwardDifferentiable]
float compute_linear_pass(float [max_size] input, TensorView<float> weight, TensorView<float> input_gradient, TensorView<float> weight_gradient, uint2 location) {
    float sum_value = 0.0;
    uint batch = location.x;
    uint pos = location.y;

    uint height = weight.size(0);
    [MaxIters(max_size)]
    for (int i = 0; i < height; i++) {
        sum_value = sum_value + input[i] * get_weight_val(weight, weight_gradient, uint2(i, pos));
    }

    return sum_value;
}

[BackwardDifferentiable]
float compute_activation(float val, no_diff int activation_type) {
    if (activation_type == 0) {
        // type 0: Identity
        return val;
    } else if (activation_type == 1) {
        // type 1: ReLU
        return max(val, 0);
    } else if (activation_type == 2) {
        // type 2: Sigmoid
    }
    return 0;
}

[BackwardDifferentiable]
float_arr mlp_compute_forward(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> input_gradient,
    TensorView<float> weight_gradients[num_layers], no_diff int activation_type[num_layers], no_diff uint batch) {

    uint input_size = input.size(1);
    float input_layer[max_size];
    float output_layer[max_size];

    [MaxIters(max_size)]
    for (int i = 0; i < input_size; i++) {
        input_layer[i] = get_input_val(input, input_gradient, uint2(batch, i));
    }
    
    [MaxIters(num_layers)]
    for (int i = 0; i < num_layers; i++) {
        uint next_height = weights[i].size(1);
        [MaxIters(max_size)]
        for (int j = 0; j < next_height; j++) {
            output_layer[j] = compute_linear_pass(input_layer, weights[i], input_gradient, weight_gradients[i], uint2(batch, j));
            output_layer[j] = compute_activation(output_layer[j], activation_type[i]);
        }
        [MaxIters(max_size)]
        for (int j = 0; j < max_size; j++) {
            input_layer[j] = output_layer[j];
        }
    }
    
    float_arr output_val;

    [MaxIters(output_size)]
    for (int i = 0; i < output_size; i++) {
        output_val.vals[i] = output_layer[i];
    }
    return output_val;
}

float get_input_val(TensorView<float> input, TensorView<float> input_gradient, uint2 location) {
    return input[location];
}

[BackwardDerivativeOf(get_input_val)]
void get_input_bwd(TensorView<float> input, TensorView<float> input_gradient, uint2(location), float derivative) {
    float old_val;
    input_gradient.InterlockedAdd(location, derivative, old_val);
}

float get_weight_val(TensorView<float> weight, TensorView<float> weight_gradient, uint2 location) {
    return weight[location];
}

[BackwardDerivativeOf(get_weight_val)]
void get_weight_bwd(TensorView<float> weight, TensorView<float> weight_gradient, uint2(location), float derivative) {
    float old_val;
    weight_gradient.InterlockedAdd(location, derivative, old_val);
}

[CudaKernel]
void mlp_fwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> output, TensorView<float> input_gradient, 
    TensorView<float> weight_gradients[num_layers], int activation_type[num_layers]) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    float_arr res = mlp_compute_forward(input, weights, input_gradient, weight_gradients, activation_type, batch);
    for (int i = 0; i < output_size; i++) {
        output[uint2(batch, i)] = res.vals[i];
    }
}

[CudaKernel]
void mlp_bwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> input_gradient, TensorView<float> weight_gradients[num_layers], 
    int activation_type[num_layers], TensorView<float> result_gradients) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    float_arr.Differential res;
    for (int i = 0; i < output_size; i++) {
        res.vals[i] = result_gradients[uint2(batch, i)];
    }
    __bwd_diff(mlp_compute_forward)(input, weights, input_gradient, weight_gradients, activation_type, batch, res);
}

[TorchEntryPoint]
TorchTensor<float> mlp_fwd(TorchTensor<float> input, TorchTensor<float> weights[num_layers], int activation_type[num_layers]) {
    var input_gradient = TorchTensor<float>.zerosLike(input);
    TensorView<float> weight_views[num_layers];
    TorchTensor<float> weight_gradients[num_layers];
    TensorView<float> weight_gradient_views[num_layers];

    [MaxIters(num_layers)]
    for (int i = 0; i < num_layers; i++) {
        weight_views[i] = weights[i].getView();
        weight_gradients[i] = TorchTensor<float>.zerosLike(weights[i]);
        weight_gradient_views[i] = weight_gradients[i].getView();
    }

    uint batches = input.size(0);
    var output = TorchTensor<float>.alloc(batches, output_size);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_fwd_kernel, blockCount, groupSize)(input, weight_views, output, input_gradient, weight_gradient_views, activation_type);
    return output;
}

[TorchEntryPoint]
tensor_arr mlp_bwd(TorchTensor<float> input, TorchTensor<float> weights[num_layers], int activation_type[num_layers], TorchTensor<float> result_gradients) {
    var input_gradient = TorchTensor<float>.zerosLike(input);
    TensorView<float> weight_views[num_layers]; 
    TorchTensor<float> weight_gradients[num_layers];
    TensorView<float> weight_gradient_views[num_layers];

    for (int i = 0; i < num_layers; i++) {
        weight_views[i] = weights[i].getView();
        weight_gradients[i] = TorchTensor<float>.zerosLike(weights[i]);
        weight_gradient_views[i] = weight_gradients[i].getView();
    }

    uint batches = input.size(0);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(input, weight_views, input_gradient, weight_gradient_views, activation_type, result_gradients);
    tensor_arr res;
    res.vals[0] = input_gradient;
    for (int i = 1; i < num_layers + 1; i++) {
        res.vals[i] = weight_gradients[i-1];
    }
    return res;
}
*/