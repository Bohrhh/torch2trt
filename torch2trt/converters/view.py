from functools import reduce
from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
def convert_view(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(output.shape)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.squeeze_')
def convert_squeeze(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None)
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]

    if dim is None:
        reduce_dim = 0
        for i, s in enumerate(input.shape):
            if s==1:
                input_trt = squeeze(ctx, input_trt, i-reduce_dim)
                reduce_dim += 1
    else:
        input_trt = squeeze(ctx, input_trt, dim)

    # get tensorrt output
    output._trt = input_trt


@tensorrt_converter('torch.unsqueeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.Tensor.unsqueeze_')
def convert_unsqueeze(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None)
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]

    # get tensorrt output
    output._trt = unsqueeze(ctx, input_trt, dim)


@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.flatten')
def convert_flatten(ctx):
    # parse args
    input     = get_arg(ctx, 'input'    , pos=0, default=None)
    start_dim = get_arg(ctx, 'start_dim', pos=1, default=0   )
    end_dim   = get_arg(ctx, 'end_dim'  , pos=2, default=-1  )
    output    = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]
    end_dim    = input.dim()-1 if end_dim==-1 else end_dim
    flatten    = input_trt.shape[start_dim:end_dim+1]
    shape_pre  = input_trt.shape[:start_dim]
    shape_post = input_trt.shape[end_dim+1:]
    if -1 in flatten:
        flatten = -1
    else:
        flatten = reduce(lambda x,y:x*y, flatten)
    shape = shape_pre+(flatten, )+shape_post
    assert sum([i==-1 for i in shape])<=1, "trt shuffle operation only support one -1 shape"    

    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = shape

    # get tensorrt output
    output._trt = layer.get_output(0)
