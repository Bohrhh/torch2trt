from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.sigmoid')
@tensorrt_converter('torch.sigmoid')
def convert_sigmoid(ctx):
    # parse args
    input = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_sigmoid_basic():
    return torch.nn.Sigmoid()