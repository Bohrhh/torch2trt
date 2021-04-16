import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{name}',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':

    setup(
        name='deform_conv',
        author='OpenMMLab',
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_ext',
                module='singleStereo.models.utils.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_conv_cuda.cpp',
                    'src/cuda/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_ext',
                module='singleStereo.models.utils.dcn',
                sources=['src/deform_pool_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_pool_cuda.cpp',
                    'src/cuda/deform_pool_cuda_kernel.cu'
                ]),

        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
