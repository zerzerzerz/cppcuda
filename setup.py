from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os

setup(
    # 安装之后在pip list里面看到的名字
    name = "myrelu",
    version = "0.0.1",
    author = "Ruizhe Zhong",

    # 把当前目录下的python packages一起进行安装
    packages = find_packages(),
    ext_modules = [
        CUDAExtension(
            # cpp package的名字，import的时候写这个
            name = "relu_cpp",
            include_dirs = [
                # 设置cpp cuda中头文件路径，注意需要是绝对路径
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relu_cppcuda/include')
            ],
            
            # 需要编译的文件，.cu和.cpp，没有.h文件
            sources = [
                "relu_cppcuda/kernel/relu_cpp.cpp",
                "relu_cppcuda/kernel/relu_cuda.cu"
            ]
        )
    ],
    cmdclass = {
        "build_ext": BuildExtension
    }
)