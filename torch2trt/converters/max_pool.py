from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.max_pool1d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.max_pool2d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.max_pool3d', enabled=trt_version() >= '7.0')
def convert_max_pool(ctx):
    # parse args
    input          = get_arg(ctx, 'input'      ,       pos=0, default=None )
    kernel_size    = get_arg(ctx, 'kernel_size',       pos=1, default=None )
    stride         = get_arg(ctx, 'stride'     ,       pos=2, default=None )
    padding        = get_arg(ctx, 'padding'    ,       pos=3, default=0    )
    dilation       = get_arg(ctx, 'dilation'    ,      pos=4, default=1    )
    ceil_mode      = get_arg(ctx, 'ceil_mode',         pos=5, default=False)
    return_indices = get_arg(ctx, 'return_indices',    pos=6, default=False)
    output         = ctx.method_return
    assert dilation==1 and return_indices==False

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    input_dim = input.dim() - 2
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim
    if not isinstance(stride, tuple):
        stride      = (stride, ) * input_dim
    if not isinstance(padding, tuple):
        padding     = (padding, ) * input_dim

    # if max_pool1d, reshape to 2D
    if input_dim == 1:
        input_trt   = unsqueeze(ctx, input_trt, -1)
        kernel_size = kernel_size + (1, )
        stride      = stride + (1, )
        padding     = padding + (0, )
    
    layer = ctx.network.add_pooling_nd(
        input=input_trt, type=trt.PoolingType.MAX, window_size=kernel_size)
    
    layer.stride_nd  = stride
    layer.padding_nd = padding
    
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    output_trt = layer.get_output(0)

    # reshape back to 1D
    if input_dim == 1:
        output_trt = squeeze(ctx, output_trt, -1)

    # get tensorrt output
    output._trt = output_trt
