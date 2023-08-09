#define max_size 64
#define output_size 3
#define num_layers 3
#define encoding_type 1 // 0: none, 1: hash, 2: frequency, 3: octree
#define group_size 128
#define samples_per_thread 1
#define resolution_base 2
#define n_levels 16
#define n_features_per_level 2
#define hashmap_size 1 << 17
#define base_resolution 16
#define per_level_scale 2.0

struct float_arr : IDifferentiable {
    float vals[output_size];
}

struct mlp_gradients {
    TorchTensor<float> gradients[2*num_layers + 2];
}

struct result_arr : IDifferentiable {
    float vals[max_size];
}


/*****************************/
/*      ENCODING METHODS     */
/*****************************/

[BackwardDifferentiable]
result_arr encoding_result(TensorView<float> input, TensorView<float> input_gradient, TensorView<float> encoding, TensorView<float> encoding_gradient, no_diff uint batch) {
    if (encoding_type == 0) {
        // no encoding, simply copy over inputs
        return no_encoding(input, input_gradient, batch);
    }
    if (encoding_type == 1) {
        // hash encoding time!
        return multiresolution_hash_encoding(input, input_gradient, encoding, encoding_gradient, batch);
    }
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

// CURRENTLY IN 2D
result_arr multiresolution_hash_encoding(TensorView<float> input, TensorView<float> input_gradient, TensorView<float> encoding, TensorView<float> encoding_gradient, no_diff uint batch) {
    result_arr output;
    // assume input values are in range 0 - 1.
    float scaled_cur = base_resolution;
    for (int i = 0; i < n_levels; i++) {
        float2 location;
        location.x = scaled_cur * get_input_val(input, input_gradient, uint2(batch, 0));
        location.y = scaled_cur * get_input_val(input, input_gradient, uint2(batch, 1));

        uint x0 = uint(floor(location.x));
        uint x1 = uint(ceil(location.x));

        uint y0 = uint(floor(location.y));
        uint y1 = uint(ceil(location.y));

        uint hashed_top_left = spatial_hash((uint(x0), uint(y0)));
        uint hashed_top_right = spatial_hash((uint(x1), uint(y0)));
        uint hashed_bot_left = spatial_hash((uint(x0), uint(y1)));
        uint hashed_bot_right = spatial_hash((uint(x1), uint(y1)));

        float frac_x = location.x - floor(location.x);
        float frac_y = location.y - floor(location.y);

        float w00 = (1 - frac_x) * (1 - frac_y);
        float w01 = (1 - frac_x) * frac_y;
        float w10 = frac_x * (1 - frac_y);
        float w11 = frac_x * frac_y;

        for (int j = 0; j < n_features_per_level; j++) {
            float feature_val = w00 * get_hash_val(encoding, encoding_gradient, uint3(hashed_top_left, i, j)) +
                                w01 * get_hash_val(encoding, encoding_gradient, uint3(hashed_top_right, i, j)) +
                                w10 * get_hash_val(encoding, encoding_gradient, uint3(hashed_bot_left, i, j)) +
                                w11 * get_hash_val(encoding, encoding_gradient, uint3(hashed_bot_right, i, j));
            output.vals[i * n_features_per_level + j] = feature_val;
        }

        scaled_cur = scaled_cur * per_level_scale;
    }

    return output;
}

[BackwardDerivativeOf(multiresolution_hash_encoding)]
void multiresolution_hash_encoding_bwd(TensorView<float> input, TensorView<float> input_gradient, TensorView<float> encoding, TensorView<float> encoding_gradient, no_diff uint batch, result_arr.Differential output_gradients) {
    // dummy method, not needed
}

// TODO: write backwards function for multiresolution hash encoding

uint spatial_hash(uint2 location) {
    uint prime_x = 1;
    uint prime_y = 2654435761;

    uint result = (prime_x * location.x ^ prime_y * location.y) % hashmap_size;
    return (uint)(result);
}

float get_hash_val(TensorView<float> encoding, TensorView<float> encoding_gradient, uint3 location) {
    return encoding[location];
}

/****************************/
/*        MLP METHODS       */
/****************************/

[BackwardDifferentiable]
result_arr mlp_forward(TensorView<float> input, TensorView<float> encoding, TensorView<float> weight[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient, TensorView<float> encoding_gradient,
                       TensorView<float> weight_gradient[num_layers], TensorView<float> bias_gradient[num_layers], no_diff int activation[num_layers], no_diff uint batch) {

    result_arr output = encoding_result(input, input_gradient, encoding, encoding_gradient, batch);

    [ForceUnroll]
    for (int i = 0; i < num_layers; i++) {
        output = linear_layer_norm(output, weight[i], bias[i], weight_gradient[i], bias_gradient[i], activation[i]);
    }

    return output;
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

    for(int i = 0; i < in_size; i++) {
        set_derivatives.vals[i] = 0.0;
    }

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
void mlp_fwd_kernel(TensorView<float> input, TensorView<float> encoding, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> output, 
    TensorView<float> input_gradient, TensorView<float> encoding_gradient, TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers]) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    if (batch > input.size(0)) {
        return;
    }

    result_arr res = mlp_forward(input, encoding, weights, bias, input_gradient, encoding_gradient, weight_gradients, bias_gradients, activation_type, batch);
    for (int i = 0; i < output_size; i++) {
        output[uint2(batch, i)] = res.vals[i];
    }
}

[CudaKernel]
void mlp_bwd_kernel(TensorView<float> input, TensorView<float> encoding, TensorView<float> weights[num_layers], TensorView<float> bias[num_layers], TensorView<float> input_gradient, 
    TensorView<float> encoding_gradient, TensorView<float> weight_gradients[num_layers], TensorView<float> bias_gradients[num_layers], int activation_type[num_layers], TensorView<float> result_gradients) {

    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    if (batch > input.size(0)) {
        return;
    }
    result_arr.Differential res;
    for (int i = 0; i < output_size; i++) {
        res.vals[i] = result_gradients[uint2(batch, i)];
    }
    __bwd_diff(mlp_forward)(input, encoding, weights, bias, input_gradient, encoding_gradient, weight_gradients, bias_gradients, activation_type, batch, res);
}

[TorchEntryPoint]
TorchTensor<float> mlp_fwd(TorchTensor<float> input, TorchTensor<float> encoding, TorchTensor<float> weights[num_layers], TorchTensor<float> bias[num_layers], int activation_type[num_layers]) {
    var input_gradient = TorchTensor<float>.zerosLike(input);
    var encoding_gradient = TorchTensor<float>.zerosLike(encoding);
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
    uint block_count = int((batches / (group_size))) + 1;
    let blockCount = uint3(block_count, 1, 1);
    let groupSize = uint3(group_size, 1, 1);
    __dispatch_kernel(mlp_fwd_kernel, blockCount, groupSize)(input, encoding, weight_views, bias_views, output, input_gradient, encoding_gradient, weight_gradient_views, bias_gradient_views, activation_type);
    return output;
}

[TorchEntryPoint]
mlp_gradients mlp_bwd(TorchTensor<float> input, TorchTensor<float> encoding, TorchTensor<float> weights[num_layers], TorchTensor<float> bias[num_layers], int activation_type[num_layers], TorchTensor<float> result_gradients) {
    var input_gradient = TorchTensor<float>.zerosLike(input);
    var encoding_gradient = TorchTensor<float>.zerosLike(encoding);
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
    
    uint block_count = int(input.size(0) / (group_size)) + 1;
    let blockCount = uint3(block_count, 1, 1);
    let groupSize = uint3(group_size, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(input, encoding, weight_views, bias_views, input_gradient, encoding_gradient, weight_gradient_views, bias_gradient_views, activation_type, result_gradients);
    mlp_gradients res;
    res.gradients[0] = input_gradient;
    res.gradients[1] = encoding_gradient;
    for (int i = 0; i < num_layers; i++) {
        res.gradients[i+2] = weight_gradients[i];
        res.gradients[i+2+num_layers] = bias_gradients[i];
    }
    return res;
}
