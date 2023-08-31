#define max_size 64
#define n_neurons 64
#define output_size 3
#define num_layers 3
#define group_size 128
#define replication_size 64
#define input_size 2
#define activation_types 1
#define wave_size 32

/*
NEED TO UPDATE SLANG TO GET THE FILL ZERO METHODS
*/

// potential issue - not loading the entire weight matrix ?


struct float_arr : IDifferentiable {
    float vals[output_size];
}

struct mlp_gradients {
    TorchTensor<float> gradients[2*num_layers + 1];
}

struct result_arr : IDifferentiable {
    float vals[max_size];
}

result_arr no_encoding(TensorView<float> input, TensorView<float> input_gradient, no_diff uint batch) {
    result_arr output;
    uint in_size = input.size(1);

    for (int i = 0; i < in_size; i++) {
        output.vals[i] = get_input_val(input, input_gradient, uint2(batch, i));
    }

    return output;
}

[BackwardDerivativeOf(no_encoding)]
void no_encoding_bwd(TensorView<float> input, TensorView<float> input_gradient, no_diff uint batch, result_arr.Differential output_gradients) {
    uint in_size = input.size(1);
    for (int i = 0; i < in_size; i++) {
        get_input_bwd(input, input_gradient, uint2(batch, i), output_gradients.vals[i]);
    }
}

[BackwardDifferentiable]
result_arr mlp_forward(TensorView<float> input, TensorView<float> weight [num_layers], TensorView<float> bias [num_layers], TensorView<float> input_gradient, 
    TensorView<float> weight_gradient [num_layers], TensorView<float> bias_gradient [num_layers], no_diff int activation [num_layers], no_diff uint batch) {
    
    result_arr output = no_encoding(input, input_gradient, batch);
    output = first_linear_layer(output, weight[0], bias[0], weight_gradient[0], bias_gradient[0], activation[0], batch);
    
    
    [ForceUnroll]
    for(int i = 1; i < num_layers-1; i++) {
        output = hidden_linear_layer(output, weight[i], bias[i], weight_gradient[i], bias_gradient[i], activation[i], input.size(0), batch);
    }

    uint layer_idx = num_layers - 1;
    output = final_linear_layer(output, weight[layer_idx], bias[layer_idx], weight_gradient[layer_idx], bias_gradient[layer_idx], activation[layer_idx], input.size(0), batch);
    
    return output;
}

result_arr first_linear_layer(result_arr input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint batch) {
    result_arr output;
    uint thread_idx = batch % group_size;
    uint in_size = input_size;
    uint out_size = n_neurons;

    groupshared float weights[input_size][n_neurons];
    groupshared float biases[n_neurons];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for(int i = 0; i < in_size; i++) {
            weights[i][thread_idx] = get_weight_val(weight, weight_gradient, uint2(i, thread_idx));
        }
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    GroupMemoryBarrierWithWaveSync();
    GroupMemoryBarrierWithGroupSync();

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for(int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.vals[j] * weights[j][i];
        }
        sum_value = sum_value + biases[i];
        output.vals[i] = compute_activation(sum_value, activation_types);
    }

    return output;
}

[BackwardDerivativeOf(first_linear_layer)]
void first_linear_layer_bwd(inout DifferentialPair<result_arr> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint batch, result_arr.Differential output_gradient) {
    uint thread_idx = batch % group_size;
    uint in_size = input_size;
    uint out_size = n_neurons;

    groupshared float weights[input_size][n_neurons];
    groupshared float biases[n_neurons];
    float preactivations[n_neurons];
    result_arr.Differential set_derivatives;

    for (int i = 0; i < in_size; i++) {
        set_derivatives.vals[i] = 0.0;
    }

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for (int i = 0; i < in_size; i++) {
            weights[i][thread_idx] = get_weight_val(weight, weight_gradient, uint2(i, thread_idx));
        }
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    GroupMemoryBarrierWithWaveSync();
    GroupMemoryBarrierWithGroupSync();

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        for(int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.p.vals[j] * weights[j][i];
        }
        sum_value = sum_value + biases[i];

        DifferentialPair<float> preactivation_value = diffPair(sum_value);
        __bwd_diff(compute_activation)(preactivation_value, activation_types, output_gradient.vals[i]);

        for(int j = 0; j < in_size; j++) {
            set_derivatives.vals[j] += weights[j][i] * preactivation_value.d;
        }
        preactivations[i] = preactivation_value.d;
    }

    if (thread_idx < n_neurons) {
        float bias_grad_bwd = WaveActiveSum(preactivations[thread_idx]);
        // get_bias_bwd(bias, bias_gradient, thread_idx, bias_grad_bwd);
        for (int j = 0; j < in_size; j++) {
            float weight_grad_bwd = WaveActiveSum(input.p.vals[j] * preactivations[thread_idx]);
            get_weight_bwd(weight, weight_gradient, uint2(j, thread_idx), weight_grad_bwd);
        }
    }
    GroupMemoryBarrierWithWaveSync();
    input = diffPair(input.p, set_derivatives);
}

result_arr hidden_linear_layer(result_arr input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint size, no_diff uint batch) {
    result_arr output;
    uint thread_idx = batch % group_size;
    uint in_size = n_neurons;
    uint out_size = n_neurons;

    groupshared float weights[n_neurons][n_neurons];
    groupshared float biases[n_neurons];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for (int i = 0; i < in_size; i++) {
            weights[i][thread_idx] = get_weight_val(weight, weight_gradient, uint2(i, thread_idx));
        }
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    GroupMemoryBarrierWithGroupSync();

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        if (batch < size) {
            for(int j = 0; j < in_size; j++) {
                sum_value = sum_value + input.vals[j] * weights[j][i];
            }
            sum_value = sum_value + biases[i];
        }
        output.vals[i] = compute_activation(sum_value, activation_types);
    }

    GroupMemoryBarrierWithWaveSync();
    return output;
}

[BackwardDerivativeOf(hidden_linear_layer)]
void hidden_linear_layer_bwd(inout DifferentialPair<result_arr> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint size, no_diff uint batch, result_arr.Differential output_gradient) {
    uint thread_idx = batch % group_size;
    uint in_size = n_neurons;
    uint out_size = n_neurons;

    result_arr.Differential set_derivatives;

    for(int i = 0; i < in_size; i++) {
        set_derivatives.vals[i] = 0.0;
    }

    groupshared float weights[n_neurons][n_neurons];
    groupshared float biases[n_neurons];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for (int i = 0; i < in_size; i++) {
            weights[i][thread_idx] = get_weight_val(weight, weight_gradient, uint2(i, thread_idx));
        }
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    GroupMemoryBarrierWithWaveSync();
    GroupMemoryBarrierWithGroupSync();

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        
        for(int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.p.vals[j] * weights[j][i];
        }
        sum_value = sum_value + biases[i];

        DifferentialPair<float> preactivation_value = diffPair(sum_value);
        __bwd_diff(compute_activation)(preactivation_value, activation_types, output_gradient.vals[i]);

        for(int j = 0; j < in_size; j++) {
            set_derivatives.vals[j] += weights[j][i] * preactivation_value.d;
        }

        // A WAVE IS NOT THE THREAD GROUP SIZE WHOOOOOPS !!! yikes
        // write to thread_idx and thread_idx + wavefrontsize = 32 all mod n_neurons
        uint loc_1 = thread_idx % n_neurons;
        uint loc_2 = (thread_idx + wave_size) % n_neurons;

        float weight_grad_bwd_loc_1 = WaveActiveSum(input.p.vals[loc_1] * preactivation_value.d);
        float weight_grad_bwd_loc_2 = WaveActiveSum(input.p.vals[loc_2] * preactivation_value.d);

        get_weight_bwd(weight, weight_gradient, uint2(loc_1, i), weight_grad_bwd_loc_1);
        get_weight_bwd(weight, weight_gradient, uint2(loc_2, i), weight_grad_bwd_loc_2);

        if (WaveIsFirstLane()) {
            float bias_grad_bwd = WaveActiveSum(preactivation_value.d);
            // get_bias_bwd(bias, bias_gradient, i, bias_grad_bwd);
        }
    }
    input = diffPair(input.p, set_derivatives);
}

result_arr final_linear_layer(result_arr input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint size, no_diff uint batch) {
    result_arr output;
    uint thread_idx = batch % group_size;
    uint in_size = n_neurons;
    uint out_size = output_size;

    groupshared float weights[n_neurons][output_size];
    groupshared float biases[output_size];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for(int i = 0; i < out_size; i++) {
            weights[thread_idx][i] = get_weight_val(weight, weight_gradient, uint2(thread_idx, i));
        }
    }
    
    
    if (thread_idx < out_size) {
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    
    //biases[thread_idx % out_size] = get_bias_val(bias, bias_gradient, thread_idx % out_size);
    GroupMemoryBarrierWithGroupSync();

    for(int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        if (batch < size) {
            for(int j = 0; j < in_size; j++) {
                sum_value = sum_value + input.vals[j] * weights[j][i];
            }
            sum_value = sum_value + biases[i];
        }
        output.vals[i] = compute_activation(sum_value, activation_types);
    }
    GroupMemoryBarrierWithWaveSync();
    return output;
}

[BackwardDerivativeOf(final_linear_layer)]
void final_linear_layer_bwd(inout DifferentialPair<result_arr> input, TensorView<float> weight, TensorView<float> bias, TensorView<float> weight_gradient, TensorView<float> bias_gradient, no_diff uint activation, no_diff uint size, no_diff uint batch, result_arr.Differential output_gradient) {
    uint thread_idx = batch % group_size;
    uint in_size = n_neurons;
    uint out_size = output_size;
    result_arr.Differential set_derivatives;

    for (int i = 0; i < in_size; i++) {
        set_derivatives.vals[i] = 0.0;
    }

    groupshared float weights[n_neurons][output_size];
    groupshared float biases[output_size];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < n_neurons) {
        for (int i = 0; i < out_size; i++) {
            weights[thread_idx][i] = get_weight_val(weight, weight_gradient, uint2(thread_idx, i));
        }
    }
    
    
    if (thread_idx < out_size) {
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    
    
    // biases[thread_idx % out_size] = get_bias_val(bias, bias_gradient, thread_idx % out_size);
    
    GroupMemoryBarrierWithGroupSync();
    AllMemoryBarrierWithWaveMaskSync(0xFFFFFFFF);

    for (int i = 0; i < out_size; i++) {
        float sum_value = 0.0;
        
        for (int j = 0; j < in_size; j++) {
            sum_value = sum_value + input.p.vals[j] * weights[j][i];
        }
        sum_value = sum_value + biases[i];
        

        DifferentialPair<float> preactivation_value = diffPair(sum_value);
        __bwd_diff(compute_activation)(preactivation_value, activation_types, output_gradient.vals[i]);

        for (int j = 0; j < in_size; j++) {
            set_derivatives.vals[j] += weights[j][i] * preactivation_value.d;
        }

        uint loc_1 = thread_idx % n_neurons;
        uint loc_2 = (thread_idx + wave_size) % n_neurons;

        float weight_grad_bwd_loc_1 = WaveActiveSum(input.p.vals[loc_1] * preactivation_value.d);
        float weight_grad_bwd_loc_2 = WaveActiveSum(input.p.vals[loc_2] * preactivation_value.d);

        get_weight_bwd(weight, weight_gradient, uint2(loc_1, i), weight_grad_bwd_loc_1);
        get_weight_bwd(weight, weight_gradient, uint2(loc_2, i), weight_grad_bwd_loc_2);
    }
    GroupMemoryBarrierWithGroupSync();
    float val = 1.0;
    float all_active = WaveActiveSum(val);
    if (WaveIsFirstLane()) {
        float bias_grad_bwd = 1;
        get_bias_bwd(bias, bias_gradient, 2, 1);
        float oldVal;
        float val = 1.0;
        // 96 (0), 32 (0), 64, 0, 1, 3
        if(batch == 3) {
            bias_gradient.InterlockedExchange(0, WaveGetLaneIndex(), oldVal);
            bias_gradient.InterlockedExchange(1, all_active, oldVal);
        }
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
    // input_gradient[location] = derivative;
    input_gradient.InterlockedAdd(location, derivative, old_val);
}

float get_weight_val(TensorView<float> weight, TensorView<float> weight_gradient, uint2 location) {
    return weight[location];
}

void get_weight_bwd(TensorView<float> weight, TensorView<float> weight_gradient, uint2 location, float derivative) {
    float old_val;
    // weight_gradient[location] = derivative;
    weight_gradient.InterlockedAdd(location, derivative, old_val);
}

float get_bias_val(TensorView<float> bias, TensorView<float> bias_gradient, uint location) {
    return bias[location];
}

void get_bias_bwd(TensorView<float> bias, TensorView<float> bias_gradient, uint location, float derivative) {
    float old_val;
    // bias_gradient[location] = derivative;
    bias_gradient.InterlockedAdd(location, derivative, old_val);
}

[CudaKernel]
void mlp_fwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> output, 
    TensorView<float> input_gradient, TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers]) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    result_arr res = mlp_forward(input, weights, bias, input_gradient, weight_gradients, bias_gradients, activation_type, batch);
    if (batch >= input.size(0))
        return;
    for (int i = 0; i < output_size; i++) {
        output[uint2(batch, i)] = res.vals[i];
    }
}

[CudaKernel]
void mlp_bwd_kernel(TensorView<float> input, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient, 
    TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers], TensorView<float> result_gradients) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;

    result_arr.Differential res;
    if (batch < input.size(0)) {
        for (int i = 0; i < output_size; i++) {
            res.vals[i] = result_gradients[uint2(batch, i)];
        }
    } else {
        for (int i = 0; i < output_size; i++) {
            res.vals[i] = 0.0;
        }
    }
    GroupMemoryBarrierWithWaveSync();
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
    let blockCount = uint3(((batches - 1) / group_size) + 1, 1, 1);
    let groupSize = uint3(group_size, 1, 1);
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
    
    for (int i = 0; i < num_layers; i++) {
        weight_views[i] = weights[i].getView();
        bias_views[i] = bias[i].getView();
        weight_gradients[i] = TorchTensor<float>.zerosLike(weights[i]);
        bias_gradients[i] = TorchTensor<float>.zerosLike(bias[i]);
        weight_gradient_views[i] = weight_gradients[i].getView();
        bias_gradient_views[i] = bias_gradients[i].getView();
    }

    uint batches = input.size(0);
    let blockCount = uint3(((batches - 1) / group_size) + 1, 1, 1);
    let groupSize = uint3(group_size, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(input, weight_views, bias_views, input_gradient, weight_gradient_views, bias_gradient_views, activation_type, result_gradients);
    mlp_gradients res;
    res.gradients[0] = input_gradient;
    for (int i = 0; i < num_layers; i++) {
        res.gradients[i+1] = weight_gradients[i];
        res.gradients[i+1+num_layers] = bias_gradients[i];
    }
    return res;
}

