from torch2trt.utils import *


def __convert_min_elementwise(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MIN)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

def __convert_min_reduce(ctx):
    # parse args
    input    = get_arg(ctx, 'input',    pos=0, default=None )
    dim      = get_arg(ctx, 'dim',      pos=1, default=None )
    keepdim  = get_arg(ctx, 'keepdim',  pos=2, default=False)
    keepdims = get_arg(ctx, 'keepdims', pos=2, default=False)
    keepdim  = keepdim or keepdims

    if dim is not None:
        output_val = ctx.method_return[0]
        output_idx = ctx.method_return[1]
    else:
        output_val = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    if dim is None:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1)
        input_trt = layer.get_output(0)

    # add tensorrt layer
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MIN, 1, torch_dim_to_trt_axes(0 if dim is None else dim, input.dim()))

    if dim is None:
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_val.reshape_dims = (1, )
        output_val._trt = layer_val.get_output(0)
    elif keepdim:
        output_val._trt = layer.get_output(0)
        output_idx._trt = layer.get_output(1)
    else:
        output_val._trt = squeeze(ctx, layer.get_output(0), dim)
        output_idx._trt = squeeze(ctx, layer.get_output(1), dim)


@tensorrt_converter('torch.min')
@tensorrt_converter('torch.Tensor.min')
def convert_min(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_min_elementwise(ctx)
    else:
        __convert_min_reduce(ctx)
        