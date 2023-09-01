#define max_size 64
#define n_neurons 64
#define output_size 3
#define group_size 64
#define input_size 64
#define wave_size 32


struct mlp_gradients {
    TorchTensor<float> gradients[1];
}

void divergence_error(TensorView<float> bias, TensorView<float> bias_gradient, no_diff uint batch) {
    uint thread_idx = batch % group_size;
    uint in_size = n_neurons;
    uint out_size = output_size;

    groupshared float biases[output_size];

    GroupMemoryBarrierWithGroupSync();
    if (thread_idx < out_size) {
        biases[thread_idx] = get_bias_val(bias, bias_gradient, thread_idx);
    }
    GroupMemoryBarrierWithGroupSync();

    float val = 1.0;
    float all_active = WaveActiveSum(val);
    if (WaveIsFirstLane()) {
        get_bias_bwd(bias, bias_gradient, 2, 1);
        float oldVal;
        // 96 (0), 32 (0), 64, 0, 1, 3
        if (batch == 3) {
            bias_gradient.InterlockedExchange(0, WaveGetLaneIndex(), oldVal);
            bias_gradient.InterlockedExchange(1, all_active, oldVal);
        }
    }
    GroupMemoryBarrierWithWaveSync();
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
void mlp_bwd_kernel(TensorView<float> bias, TensorView<float> bias_gradients) {
    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    divergence_error(bias, bias_gradients, batch);
}

[TorchEntryPoint]
mlp_gradients mlp_bwd(TorchTensor<float> bias) {
    TorchTensor<float> bias_gradient = TorchTensor<float>.zerosLike(bias);

    uint batches = 64;
    let blockCount = uint3(1, 1, 1);
    let groupSize = uint3(64, 1, 1);
    __dispatch_kernel(mlp_bwd_kernel, blockCount, groupSize)(bias, bias_gradient);
    mlp_gradients res;
    res.gradients[0] = bias_gradient;

    return res;
}
