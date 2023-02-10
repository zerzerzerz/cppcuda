#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIV_UP(n,x) ((n-1+x)) / x
const dim3 block(16,16);

template <typename scalar_t>
__global__ void relu_kernel_forward(
    // 32表示index是32位的
    // scalar_t是模板参数
    // 2是tensor有几个维度
    // torch::RestrictPtrTraits对应const __restrict__ *
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output
){
    const size_t b = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if((b < input.size(0)) && (f < input.size(1))){
        output[b][f] = (input[b][f]>0)? input[b][f]: scalar_t(0);
    }
    return;
}


torch::Tensor relu_cuda_forward(const torch::Tensor& input){
    auto output = torch::zeros_like(input);
    const dim3 grid(DIV_UP(input.size(1), block.x), DIV_UP(input.size(0), block.y));

    // 动态类型分配，根据输入的类型自动确定模板参数
    // 第一个参数是输入的类型
    // 第二个参数是error字符串
    // 第三个参数是调用的匿名函数(lambda表达式)
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_cuda_forward", ([&] {
        // <scalar_t>告诉kernel模板参数，在AT_DISPATCH_FLOATING_TYPES下只能为这个名字！！！
        // 不需要写template <typename scalar_t>
        // 其实是个#define
        relu_kernel_forward<scalar_t><<<grid,block>>>(
            // 从tensor到accessor，注意这是个函数，要加()来调用
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return output;
}



template <typename scalar_t>
__global__ void relu_kernel_backward(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input
){
    const size_t b = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if((b<grad_output.size(0)) && (f<grad_output.size(1))){
        grad_input[b][f] = (output[b][f]>0)? grad_output[b][f]: scalar_t(0);
    }
    return;
}


torch::Tensor relu_cuda_backward(const torch::Tensor& grad_output, const torch::Tensor& output){
    auto grad_input = torch::zeros_like(grad_output);
    const dim3 grid(DIV_UP(output.size(1), block.x), DIV_UP(output.size(0), block.y));

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "relu_cuda_backward", ([&]{
        relu_kernel_backward<scalar_t><<<grid, block>>>(
            grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return grad_input;
}