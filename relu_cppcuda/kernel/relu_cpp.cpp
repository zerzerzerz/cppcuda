#include <torch/extension.h>
#include "utils.h"

torch::Tensor relu_cpp_forward(const torch::Tensor& input){
    CHECK_INPUT(input);
    return relu_cuda_forward(input);
}

torch::Tensor relu_cpp_backward(const torch::Tensor& grad_output, const torch::Tensor& output){
    CHECK_INPUT(grad_output);
    CHECK_INPUT(output);
    return relu_cuda_backward(grad_output, output);
}

// 绑定cpp和py
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // 第一个参数指定python怎么调用
    // 第二个参数为需要绑定的函数地址
    // 第三个字符串是说明字符串
    m.def("relu_cpp_forward", &relu_cpp_forward, "relu_cpp_forward");
    m.def("relu_cpp_backward", &relu_cpp_backward, "relu_cpp_backward");
}