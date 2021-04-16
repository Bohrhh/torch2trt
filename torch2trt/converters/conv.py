from torch2trt.utils import *

@tensorrt_converter('torch.nn.functional.conv1d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv2d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv3d', enabled=trt_version() >= '7.0')
def convert_conv_function(ctx):
    # parse args
    input        = ctx.method_args[0]
    weight       = ctx.method_args[1]
    kernel_size  = tuple(weight.shape[2:])
    out_channels = weight.shape[0]
    bias         = get_arg(ctx, 'bias'    , pos=2, default=None)
    stride       = get_arg(ctx, 'stride'  , pos=3, default=1)
    padding      = get_arg(ctx, 'padding' , pos=4, default=0)
    dilation     = get_arg(ctx, 'dilation', pos=5, default=1)
    groups       = get_arg(ctx, 'groups'  , pos=6, default=1)

    kernel       = weight.detach().cpu().numpy()
    bias         = bias.detach().cpu().numpy() if bias is not None else None
    output       = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    input_dim = input.dim() - 2
    if not isinstance(kernel_size, (tuple, list)):
        kernel_size = (kernel_size, ) * input_dim
    if not isinstance(stride, (tuple, list)):
        stride      = (stride, ) * input_dim
    if not isinstance(padding, (tuple, list)):
        padding     = (padding, ) * input_dim
    if not isinstance(dilation, (tuple, list)):
        dilation    = (dilation, ) * input_dim

    # if conv1d, reshape to 2D
    if input_dim == 1:
        input_trt   = unsqueeze(ctx, input_trt, -1)
        kernel_size = kernel_size + (1, )
        stride      = stride + (1, )
        padding     = padding + (0, )
        dilation    = dilation + (1, )
        kernel      = kernel[..., None]

    layer = ctx.network.add_convolution_nd(
        input=input_trt,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd   = stride
    layer.padding_nd  = padding
    layer.dilation_nd = dilation

    if groups is not None:
        layer.num_groups = groups

    output_trt = layer.get_output(0)

    # reshape back to 1D
    if input_dim == 1:
        output_trt = squeeze(ctx, output_trt, -1)

    output._trt = output_trt