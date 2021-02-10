from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    # parse args
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=0   ) 
    output = ctx.method_return

    # get tensorrt input
    trt_inputs = add_missing_trt_tensors(ctx.network, inputs)
    trt_inputs = broadcast_trt_tensors(ctx.network, trt_inputs, len(output.shape))

    # add tensorrt layer
    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_cat_basic():
    return TestInterface(lambda *x: torch.cat(x, dim=1))
