from torch2trt.utils import *


@tensorrt_converter('torch.topk')
def convert_topk(ctx):
    # parse args
    input   = get_arg(ctx, 'input',   pos=0, default=None)
    k       = get_arg(ctx, 'k',       pos=1, default=None)
    dim     = get_arg(ctx, 'dim',     pos=2, default=None)
    largest = get_arg(ctx, 'largest', pos=3, default=True)
    sorted  = get_arg(ctx, 'sorted',  pos=4, default=True)
    assert input is not None and k is not None
    assert k<=1024, 'Currently only values of K up to 1024 are supported.'
    assert sorted,  'Currently only sorted results are supported.'
    dim = -1 if dim is None else dim

    output_val = ctx.method_return[0]
    output_idx = ctx.method_return[1]

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    if input.dim()==1:
        dim = 1
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (1, -1)
        input_trt = layer.get_output(0)
    
    op = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN
    layer = ctx.network.add_topk(input_trt, op, k, torch_dim_to_trt_axes(dim, input.dim()))

    # get tensorrt output

    if input.dim()==1:
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_val.reshape_dims = (-1, )
        output_val._trt = layer_val.get_output(0)
        layer_idx = ctx.network.add_shuffle(layer.get_output(1))
        layer_idx.reshape_dims = (-1, )
        output_idx._trt = layer_idx.get_output(0)
    else:
        output_val._trt = layer.get_output(0)
        output_idx._trt = layer.get_output(1)