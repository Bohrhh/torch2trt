import sys
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

ext_modules = []
    
setup(
    name='torch2trt',
    version='0.1.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    ext_package='torch2trt',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
