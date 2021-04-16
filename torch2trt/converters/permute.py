from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.permute')
def convert_permute(ctx):
    # parse args
    input  = ctx.method_args[0]
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c, d)
    else:
        permutation = tuple(ctx.method_args[1])   # handle permute([a, b, c, d])
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
   
    # get tensorrt output
    output._trt = layer.get_output(0)
