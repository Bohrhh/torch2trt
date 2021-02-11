import numpy as np
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *

@tensorrt_converter('torch.nn.functional.batch_norm', enabled=trt_version() >= '7.0')
def convert_batch_norm(ctx):
    # parse args
    input        = get_arg(ctx, 'input'       , pos=0, default=None) 
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None) 
    running_var  = get_arg(ctx, 'running_var' , pos=2, default=None) 
    weight       = get_arg(ctx, 'weight'      , pos=3, default=None) 
    bias         = get_arg(ctx, 'bias'        , pos=4, default=None) 
    eps          = get_arg(ctx, 'eps'         , pos=7, default=10e-6) 
    output       = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    input_dim = input.dim() - 2
    
    scale = weight.detach().cpu().numpy() / np.sqrt(running_var.detach().cpu().numpy() + eps)
    bias  = bias.detach().cpu().numpy() - running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    # if batch norm 1d, reshape to 2D
    if input_dim == 1:
        layer              = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = tuple(input.shape)+(1,)
        input_trt          = layer.get_output(0)
        scale              = scale[..., None]
        power              = power[..., None]

    layer = ctx.network.add_scale_nd(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 1)

    # reshape back to 1D
    if input_dim == 1:
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = output.shape

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)], enabled=trt_version() >= '7.0')
def test_batch_norm_1d():
    return torch.nn.BatchNorm1d(10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4)], enabled=trt_version() >= '7.0')
def test_batch_norm_2d():
    return torch.nn.BatchNorm2d(10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4, 5)], enabled=trt_version() >= '7.0')
def test_batch_norm_3d():
    return torch.nn.BatchNorm3d(10)
    