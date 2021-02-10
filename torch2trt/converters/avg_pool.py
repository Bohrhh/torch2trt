from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *

@tensorrt_converter('torch.nn.functional.avg_pool1d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.avg_pool2d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.avg_pool3d', enabled=trt_version() >= '7.0')
def convert_avg_pool_trt7(ctx):
    # parse args
    input             = get_arg(ctx, 'input'      ,       pos=0, default=None )
    kernel_size       = get_arg(ctx, 'kernel_size',       pos=1, default=None )
    stride            = get_arg(ctx, 'stride'     ,       pos=2, default=None )
    padding           = get_arg(ctx, 'padding'    ,       pos=3, default=0    )
    ceil_mode         = get_arg(ctx, 'ceil_mode'  ,       pos=4, default=False)
    count_include_pad = get_arg(ctx, 'count_include_pad', pos=5, default=True )
    output            = ctx.method_return

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

    # if avg_pool1d, reshape to 2D
    if input_dim == 1:
        layer              = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = tuple(input.shape)+(1,)
        input_trt          = layer.get_output(0)
        kernel_size        = kernel_size + (1, )
        stride             = stride + (1, )
        padding            = padding + (0, )
    
    layer = ctx.network.add_pooling_nd(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    
    layer.stride_nd  = stride
    layer.padding_nd = padding
    layer.average_count_excludes_padding = not count_include_pad
    
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    # reshape back to 1D
    if input_dim == 1:
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = output.shape

    # get tensorrt output
    output._trt = layer.get_output(0)
 

# =========================================
# test avg_pool1d 
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0')
def test_avg_pool1d_k1s1p0():
    return torch.nn.AvgPool1d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0')
def test_avg_pool1d_k3s1p0():
    return torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0')
def test_avg_pool1d_k3s2p0():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0')
def test_avg_pool1d_k3s2p1():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0')
def test_avg_pool1d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)

# =========================================
# test avg_pool2d
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0')
def test_avg_pool2d_k1s1p0():
    return torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0')
def test_avg_pool2d_k3s1p0():
    return torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0')
def test_avg_pool2d_k3s2p0():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0')
def test_avg_pool2d_k3s2p1():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0')
def test_avg_pool2d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)


# =========================================
# test avg_pool3d
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0')
def test_avg_pool3d_k1s1p0():
    return torch.nn.AvgPool3d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0')
def test_avg_pool3d_k3s1p0():
    return torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0')
def test_avg_pool3d_k3s2p0():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0')
def test_avg_pool3d_k3s2p1():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0')
def test_avg_pool3d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)


