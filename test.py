import torch
import relu_py
import time

batch_size = 1024 + 1
num_feature = 2048 + 1

print("warming up")
for _ in range(10):
    tmp = torch.randn(batch_size, num_feature).cuda().requires_grad_()
    tmp = relu_py.relu(tmp) * 2
    loss_cuda = tmp.mean()
    loss_cuda.backward()


a_cuda = torch.randn(batch_size, num_feature).cuda().requires_grad_()
torch.cuda.synchronize()
s = time.time()
b_cuda = relu_py.relu(a_cuda) * 2
loss_cuda = b_cuda.mean()
loss_cuda.backward()
torch.cuda.synchronize()
e = time.time()
print("time for  cuda is {:.6f}".format(e-s))



a_torch = a_cuda.detach().clone().requires_grad_()
torch.cuda.synchronize()
s = time.time()
b_torch = torch.nn.functional.relu(a_torch) * 2
loss_torch = b_torch.mean()
loss_torch.backward()
torch.cuda.synchronize()
e = time.time()
print("time for torch is {:.6f}".format(e-s))


print("value equal?", torch.allclose(b_cuda, b_torch))
print("grad equal?", torch.allclose(a_cuda.grad, a_torch.grad))