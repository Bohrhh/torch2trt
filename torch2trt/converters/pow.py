from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.pow')
@tensorrt_converter('torch.Tensor.__ipow__')
@tensorrt_converter('torch.Tensor.__pow__')
def convert_pow(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)

    # get tensorrt output
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rpow__')
def convert_rpow(ctx):
    # parse args
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_pow_basic():
    return TestInterface(lambda x, y: x**y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_pow():
    return TestInterface(lambda x, y: torch.pow(x,y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_int():
    return TestInterface(lambda x: 2**x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_float():
    return TestInterface(lambda x: 2.0**x)
