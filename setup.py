import os
import sys
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

ext_modules = []
data_files = []

if '--plugins' in sys.argv:
    dcn_sos = [os.path.join('torch2trt/opts/dcn', f) for f in os.listdir('torch2trt/opts/dcn') if f.endswith(".so")]
    sys.argv.remove('--plugins')
    data_files = [('torch2trt/opts/dcn', dcn_sos)]

    
setup(
    name='torch2trt',
    version='0.1.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    ext_package='torch2trt',
    ext_modules=ext_modules,
    data_files=data_files,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
