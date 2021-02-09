import numpy as np
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.module_test import add_module_test
from torch2trt.utils import *

@tensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
def convert_adaptive_avg_pool2d(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    output_size = ctx.method_args[1]
    if isinstance(output_size, int):
        output_size = (output_size, ) * 2

    kernel_size = (int(np.ceil(float(input.shape[2])/output_size[0])), 
                   int(np.ceil(float(input.shape[3])/output_size[1])))
    
    stride = ((input.shape[2]-kernel_size[0])//(output_size[0]-1) if output_size[0]>1 else 1,
              (input.shape[3]-kernel_size[1])//(output_size[1]-1) if output_size[1]>1 else 1)
    
    assert stride[0]*(output_size[0]-1)+kernel_size[0]==input.shape[2], \
        "Input width:{}, output width:{} would make trt kernel size inconsistent.".format(input.shape[2], output_size[0])
    assert stride[1]*(output_size[1]-1)+kernel_size[1]==input.shape[3], \
        "Input height:{}, output height:{} would make trt kernel size inconsistent.".format(input.shape[2], output_size[0])

    layer = ctx.network.add_pooling(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_avg_pool2d_1x1():
    return torch.nn.AdaptiveAvgPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_avg_pool2d_2x2():
    return torch.nn.AdaptiveAvgPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 7, 7)])
def test_adaptive_avg_pool2d_3x3():
    return torch.nn.AdaptiveAvgPool2d((3, 3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 13, 13)])
def test_adaptive_avg_pool2d_4x4():
    return torch.nn.AdaptiveAvgPool2d((4, 4))