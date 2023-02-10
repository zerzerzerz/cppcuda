#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


torch::Tensor relu_cuda_forward(const torch::Tensor& input);
torch::Tensor relu_cuda_backward(const torch::Tensor& grad_output, const torch::Tensor& output);