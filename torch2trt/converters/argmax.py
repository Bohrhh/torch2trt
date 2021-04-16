from torch2trt.utils import *
    
    
@tensorrt_converter('torch.argmax')
@tensorrt_converter('torch.Tensor.argmax')
def convert_argmax(ctx):
    # parse args
    input    = ctx.method_args[0]
    dim      = get_arg(ctx, 'dim',      pos=1, default=None)
    keepdim  = get_arg(ctx, 'keepdim',  pos=2, default=False)
    keepdims = get_arg(ctx, 'keepdims', pos=2, default=False)
    keepdim  = keepdim or keepdims
    output   = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    if dim is None:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1)
        input_trt = layer.get_output(0)
    
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MAX, 1, torch_dim_to_trt_axes(0 if dim is None else dim, input.dim()))

    if dim is None:
        layer = ctx.network.add_shuffle(layer.get_output(1))
        layer.reshape_dims = (1, )
        output._trt = layer.get_output(0)
    elif keepdim:
        output._trt = layer.get_output(1)
    else:
        output_trt  = layer.get_output(1)
        output._trt = squeeze(ctx, output_trt, dim)
        