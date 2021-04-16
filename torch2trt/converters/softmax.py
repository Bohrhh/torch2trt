from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return
    assert dim is not None, 'Dim should be provided!'
    # assert dim != 0, 'There is large error in test when dim is zero.'

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = torch_dim_to_trt_axes(dim, input.dim())

    # get tensorrt output
    output._trt = layer.get_output(0)
