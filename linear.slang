[BackwardDifferentiable]
float computeOutputValue(TensorView<float> input, TensorView<float> weight, TensorView<float> inputGradToPropogateTo, 
    TensorView<float> weightGradToPropogateTo, uint location)
{
    float sumValue = 0.0;
    uint height = weight.size(0);
    uint width = weight.size(1);
    if (location >= width || location < 0)
        return 0;
    
    float pow = 1;

    [MaxIters (10000)]
    for (int i = 0; i < height; i++) {
        sumValue = sumValue + getWeightElement(weight, weightGradToPropogateTo, uint2(i, location)) * getInputElement(input, inputGradToPropogateTo, i);
    }
    return sumValue;
}

[CudaKernel]
void linear_fwd_kernel(TensorView<float> input, TensorView<float> weight, TensorView<float> output, 
    TensorView<float> inputGradToPropogateTo, TensorView<float> weightGradToPropogateTo) 
{
    uint location = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).x;

    if (location > input.size(0))
        return;

    output[location] = computeOutputValue(input, weight, inputGradToPropogateTo, weightGradToPropogateTo, location);
}

[TorchEntryPoint]
TorchTensor<float> linear_fwd(TorchTensor<float> input, TorchTensor<float> weight)
{
    var result = TorchTensor<float>.alloc(weight.size(1));
    let blockCount = uint3(1);
    let groupSize = uint3(weight.size(0), 1, 1);
    var inputGradToPropogateTo = TorchTensor<float>.zerosLike(input);
    var weightGradToPropogateTo = TorchTensor<float>.zerosLike(weight);

    __dispatch_kernel(linear_fwd_kernel, blockCount, groupSize)(input, weight, result, inputGradToPropogateTo, weightGradToPropogateTo);
    return result;
}


[CudaKernel]
void linear_bwd_kernel(TensorView<float> input, TensorView<float> weight, TensorView<float> inputGradToPropogateTo,
    TensorView<float> weightGradToPropogateTo, TensorView<float> resultGradToPropogateFrom)
{
    uint location = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).x;

    if (location > input.size(0))
        return;

    float result = resultGradToPropogateFrom[location];
    __bwd_diff(computeOutputValue)(input, weight, inputGradToPropogateTo, weightGradToPropogateTo, location, result);

}

struct TwoTensors
{
    TorchTensor<float> tensors[2];
}

[TorchEntryPoint]
TorchTensor<float> linear_bwd_input(TorchTensor<float> input, TorchTensor<float> weight, TorchTensor<float> resultGradToPropogateFrom) {
    var result = TorchTensor<float>.alloc(weight.size(1));
    let blockCount = uint3(1);
    let groupSize = uint3(input.size(0), 1, 1);
    var inputGradToPropogateTo = TorchTensor<float>.zerosLike(input);
    var weightGradToPropogateTo = TorchTensor<float>.zerosLike(weight);
    __dispatch_kernel(linear_bwd_kernel, blockCount, groupSize)(input, weight, inputGradToPropogateTo, weightGradToPropogateTo, resultGradToPropogateFrom);
    return inputGradToPropogateTo;
}

[TorchEntryPoint]
TorchTensor<float> linear_bwd_weight(TorchTensor<float> input, TorchTensor<float> weight, TorchTensor<float> resultGradToPropogateFrom) {
    var result = TorchTensor<float>.alloc(weight.size(1));
    let blockCount = uint3(1);
    let groupSize = uint3(input.size(0), 1, 1);
    var inputGradToPropogateTo = TorchTensor<float>.zerosLike(input);
    var weightGradToPropogateTo = TorchTensor<float>.zerosLike(weight);
    __dispatch_kernel(linear_bwd_kernel, blockCount, groupSize)(input, weight, inputGradToPropogateTo, weightGradToPropogateTo, resultGradToPropogateFrom);
    return weightGradToPropogateTo;
}


float getInputElement(TensorView<float> input, TensorView<float> inputGradToPropogateTo, uint location) {
    return input[location];
}

[BackwardDerivativeOf(getInputElement)]
void getInputElement_bwd(TensorView<float> input, TensorView<float> inputGradToPropogateTo, uint location, float derivative) {
    float oldVal;
    inputGradToPropogateTo.InterlockedAdd(location, derivative, oldVal);
}

float getWeightElement(TensorView<float> weight, TensorView<float> weightGradToPropogateTo, uint2 location) {
    return weight[location.xy];
}

[BackwardDerivativeOf(getWeightElement)]
void getWeightElement_bwd(TensorView<float> weight, TensorView<float> weightGradToPropogateTo, uint2 location, float derivative) {
    float oldVal;
    weightGradToPropogateTo.InterlockedAdd(location, derivative, oldVal);
}


