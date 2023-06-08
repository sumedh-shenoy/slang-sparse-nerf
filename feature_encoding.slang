[BackwardDifferentiable]
float bilinearInterpolation(TensorView<float> feature_grid, TensorView<float> feature_gradient, float2 loc, uint dim) {
    float x0 = floor(loc.x);
    float y0 = floor(loc.y);
    float x1 = x0 + 1;
    float y1 = y0 + 1;

    float fracX = loc.x - floor(loc.x);
    float fracY = loc.y - floor(loc.y);

    float w00 = (1 - fracX) * (1 - fracY);
    float w01 = (1 - fracX) * fracY;
    float w10 = fracX * (1 - fracY);
    float w11 = fracX * fracY;

    float value = w00 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x0), int(y0), dim)) +
                  w01 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x0), int(y0), dim)) +
                  w10 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x1), int(y0), dim)) +
                  w11 * getFeatureElement(feature_grid, feature_gradient, uint3(int(x1), int(y1), dim));

    return value;
}

[CudaKernel]
void feature_encoding_fwd_kernel(TensorView<float> feature_grid, TensorView<float> feature_gradient, TensorView<float> output, float2 loc) {
    uint dim = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).x;
    float val = bilinearInterpolation(feature_grid, feature_gradient, loc, dim);
    output[dim] = val;
}

[TorchEntryPoint]
TorchTensor<float> feature_encoding_fwd(TorchTensor<float> feature_grid, float2 loc) {
    uint dim = 2;
    float rescale = 4;
    float2 uv = loc / rescale;

    var result = TorchTensor<float>.alloc(2);
    let blockCount = uint3(1);
    let groupSize = uint3(dim, 1, 1);
    var feature_gradient = TorchTensor<float>.zerosLike(feature_grid);

    __dispatch_kernel(feature_encoding_fwd_kernel, blockCount, groupSize)(feature_grid, feature_gradient, result, loc);
    return result;
}

void feature_encoding_bwd_kernel(TensorView<float> feature_grid, TensorView<float> feature_gradient, TensorView<float> result_gradient) {
    
}
 

float getFeatureElement(TensorView<float> feature_grid, TensorView<float> feature_gradient, uint3 loc) {
    return feature_grid[loc];
}

[BackwardDerivativeOf(getFeatureElement)]
void getFeatureElement_bwd(TensorView<float> feature_grid, TensorView<float> feature_gradient, uint3 loc, float derivative) {
    float oldVal;
    feature_gradient.InterlockedAdd(loc, derivative, oldVal);
}


