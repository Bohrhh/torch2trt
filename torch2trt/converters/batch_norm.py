import numpy as np
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
        input_trt = unsqueeze(ctx, input_trt, -1)
        scale     = scale[..., None]
        power     = power[..., None]

    layer = ctx.network.add_scale_nd(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 1)
    output_trt = layer.get_output(0)

    # reshape back to 1D
    if input_dim == 1:
        output_trt = squeeze(ctx, output_trt, -1)

    # get tensorrt output
    output._trt = output_trt


