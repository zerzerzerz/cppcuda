import torch
import relu_cpp
from torch.autograd import Function

# 对cpp函数进行封装，之后就可以完全按照pytorch原生的方式来操作了
class ReLUFunction(Function):
    # 声明为静态方法
    @staticmethod
    # ctx是上下文变量管理器，比如计算梯度的时候需要中间变量，就存到这里面
    # 后面的参数就是输出
    def forward(ctx, input):
        output = relu_cpp.relu_cpp_forward(input)
        # 加入上下文变量管理器
        ctx.save_for_backward(output)
        return output

    @staticmethod
    # 第一个参数是上下文管理器
    # 后面的参数要依次对应forward的输出的梯度
    # 返回值需要依次对应forward的输入的梯度
    def backward(ctx, grad_output):
        grad_input = relu_cpp.relu_cpp_backward(grad_output, *ctx.saved_tensors)
        return grad_input


# 导出这个函数让其他地方调用，调用的时候就按照forward来操作就行
relu = ReLUFunction.apply