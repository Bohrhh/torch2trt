from torch2trt.utils import *


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx):
    # parse args
    input    = get_arg(ctx, 'input',    pos=0, default=None )
    dim      = get_arg(ctx, 'dim',      pos=1, default=tuple(range(input.dim())))
    keepdim  = get_arg(ctx, 'keepdim',  pos=2, default=False)
    keepdims = get_arg(ctx, 'keepdims', pos=2, default=False)
    keepdim  = keepdim or keepdims
    output   = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, torch_dim_to_trt_axes(dim, input.dim()), keepdim)

    # get tensorrt output
    output._trt = layer.get_output(0)
    