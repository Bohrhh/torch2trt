from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.relu6')
def convert_functional_relu6(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input, 6])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    layer = ctx.network.add_activation(input=input_a_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu6_basic():
    return torch.nn.ReLU6()    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu6_basic():
    return TestInterface(lambda x: torch.nn.functional.relu6(x))