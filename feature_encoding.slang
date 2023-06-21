#define num_features 4
#define rescaling 4

struct float_arr : IDifferentiable {
    float vals[num_features];
}

[BackwardDifferentiable]
float_arr bilinearInterpolation(TensorView<float> feature_grid, TensorView<float> feature_gradient, no_diff float2 loc) {
    float x0 = floor(loc.x);
    float y0 = floor(loc.y);
    float x1 = x0 + 1;
    float y1 = y0 + 1;

    float fracX = loc.x - no_diff floor(loc.x);
    float fracY = loc.y - no_diff floor(loc.y);

    float w00 = (1 - fracX) * (1 - fracY);
    float w01 = (1 - fracX) * fracY;
    float w10 = fracX * (1 - fracY);
    float w11 = fracX * fracY;

    float_arr val;

    // with multiple features (not multiresolution hashing)
    [MaxIters(num_features)]
    for (int i = 0; i < num_features; i++) {
        float feature_val = w00 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x0), int(y0), i)) +
                            w01 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x0), int(y0), i)) +
                            w10 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x1), int(y0), i)) +
                            w11 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x1), int(y1), i));
        val.vals[i] = feature_val;
    }

    return val;
}

// (x, y, pos)
float getFeatureElement(TensorView<float> feature_grid, TensorView<float> feature_gradient, no_diff uint3 loc) {
    return feature_grid[loc];
}

[BackwardDerivativeOf(getFeatureElement)]
void getFeatureElement_bwd(TensorView<float> feature_grid, TensorView<float> feature_gradient, no_diff uint3 loc, float derivative) {
    float oldVal;
    feature_gradient.InterlockedAdd(loc, derivative, oldVal);
}

[CudaKernel]
void feature_encoding_fwd_kernel(TensorView<float> input, TensorView<float> feature_grid, TensorView<float> feature_gradient, TensorView<float> output) {
    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    float loc_x = input[uint2(batch, 0)] / rescaling;
    float loc_y = input[uint2(batch, 1)] / rescaling;
    float2 loc = (loc_x, loc_y);

    float_arr val = bilinearInterpolation(feature_grid, feature_gradient, loc);
    for(int i = 0; i < num_features; i++) {
        output[uint2(batch, i)] = val.vals[i];
    }
}

[CudaKernel]
void feature_encoding_bwd_kernel(TensorView<float> input, TensorView<float> feature_grid, TensorView<float> feature_gradient, TensorView<float> result_gradient) {
    uint batch = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;

    float loc_x = input[uint2(batch, 0)] / rescaling;
    float loc_y = input[uint2(batch, 1)] / rescaling;
    float2 loc = (loc_x, loc_y);

    float_arr.Differential res;
    for (int i = 0; i < num_features; i++) {
        res.vals[i] = result_gradient[uint2(batch, i)];
    }
    __bwd_diff(bilinearInterpolation)(feature_grid, feature_gradient, loc, res);
}

[TorchEntryPoint]
TorchTensor<float> feature_encoding_fwd(TorchTensor<float> feature_grid, TorchTensor<float> locs) {
    uint batches = locs.size(0);

    var result = TorchTensor<float>.alloc(batches, num_features);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    var feature_gradient = TorchTensor<float>.zerosLike(feature_grid);

    __dispatch_kernel(feature_encoding_fwd_kernel, blockCount, groupSize)(locs, feature_grid, feature_gradient, result);
    return result;
}

[TorchEntryPoint]
TorchTensor<float> feature_encoding_bwd(TorchTensor<float> feature_grid, TorchTensor<float> locs, TorchTensor<float> result_gradient) {
    var feature_gradient = TorchTensor<float>.zerosLike(feature_grid);

    uint batches = locs.size(0);
    let blockCount = uint3(1);
    let groupSize = uint3(batches, 1, 1);
    __dispatch_kernel(feature_encoding_bwd_kernel, blockCount, groupSize)(locs, feature_grid, feature_gradient, result_gradient);
    return feature_gradient;
}
