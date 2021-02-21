from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_shape(input=input_trt)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_size_basic():
    return TestInterface(lambda x: x.size())