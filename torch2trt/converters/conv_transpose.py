from torch2trt.utils import *

@tensorrt_converter('torch.nn.ConvTranspose1d.forward', enabled=trt_version() >= '7.1.3')
@tensorrt_converter('torch.nn.ConvTranspose2d.forward', enabled=trt_version() >= '7.1.3')
@tensorrt_converter('torch.nn.ConvTranspose3d.forward', enabled=trt_version() >= '7.1.3')
def convert_ConvTranspose2d_trt7(ctx):
    # parse args
    module      = ctx.method_args[0]
    input       = ctx.method_args[1]
    kernel_size = module.kernel_size
    stride      = module.stride
    padding     = module.padding
    dilation    = module.dilation
    groups      = module.groups
    kernel      = module.weight.detach().cpu().numpy()
    bias        = module.bias.detach().cpu().numpy() if module.bias is not None else None
    output      = ctx.method_return

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
    if not isinstance(dilation, tuple):
        dilation    = (dilation, ) * input_dim

    # if conv1d, reshape to 2D
    if input_dim == 1:
        input_trt   = unsqueeze(ctx, input_trt, -1)
        kernel_size = kernel_size + (1, )
        stride      = stride + (1, )
        padding     = padding + (0, )
        dilation    = dilation + (1, )
        kernel      = kernel[..., None]

    layer = ctx.network.add_deconvolution_nd(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    
    if groups is not None:
        layer.num_groups = groups

    output_trt = layer.get_output(0)

    # reshape back to 1D
    if input_dim == 1:
        output_trt = squeeze(ctx, output_trt, -1)

    output._trt = output_trt



# =========================================
# test ConvTranspose1d 
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k1s1p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k3s1p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k3s2p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k3s2p1d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k3s2p1d2():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose1d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', dynamic_axes={0:[1,32], 2:[128,256]})
def test_ConvTranspose1d_k1s1p0d1_dynamic():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

# =========================================
# test ConvTranspose2d 
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k1s1p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k3s1p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k3s2p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k3s2p1d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k3s2p1d2():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose2d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_ConvTranspose2d_k1s1p0d1_dynamic():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

# =========================================
# test ConvTranspose3d 
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k1s1p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k3s1p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k3s2p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k3s2p1d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k3s2p1d2():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3')
def test_ConvTranspose3d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', dynamic_axes={0:[1,32], 2:[64,100], 3:[64,100], 4:[64,100]})
def test_ConvTranspose3d_k1s1p0d1_dynamic():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)