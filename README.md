# README
```bash
pip install -e .
python test.py
```
- `relu_py/relu_py.py`
  - 封装了pytorch-python，能够和原生pytorch一样调用反向传播
  - 调用用cpp实现的python函数
- `relu_cppcuda/kernel/relu_cpp.cpp`
  - 封装了python-cpp，完成这一步就可以import了
  - 调用host function
- `relu_cppcuda/kernel/relu_cuda.cu`
  - 包含kernel和host
  - host调用kernel