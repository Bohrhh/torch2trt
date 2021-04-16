from torch2trt.utils import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    # parse args
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=0   ) 
    output = ctx.method_return

    # get tensorrt input
    inputs_trt = add_missing_trt_tensors(ctx.network, inputs)
    inputs_trt = broadcast_trt_tensors(ctx.network, inputs_trt, output.dim())

    # add tensorrt layer
    layer = ctx.network.add_concatenation(inputs=inputs_trt)
    layer.axis = dim

    # get tensorrt output
    output._trt = layer.get_output(0)
