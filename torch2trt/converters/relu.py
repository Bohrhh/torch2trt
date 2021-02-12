from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
def convert_functional_relu(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input=input_trt, type=trt.ActivationType.RELU)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu_basic():
    return torch.nn.ReLU()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu_basic():
    return TestInterface(lambda x: torch.nn.functional.relu(x))