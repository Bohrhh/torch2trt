import torch.nn as nn
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.normalize')
def convert_normalize(ctx):
    # get args
    input  = get_arg(ctx, 'input', pos=0, default=None )
    p      = get_arg(ctx, 'p',     pos=1, default=2.0  )
    dim    = get_arg(ctx, 'dim',   pos=2, default=1    )
    eps    = get_arg(ctx, 'eps',   pos=3, default=1e-12)
    output = ctx.method_return

    p = float(p)
    eps = float(eps)

    # get tensorrt input
    input_trt, eps_trt, p_trt, p_inv_trt = add_missing_trt_tensors(ctx.network, [input, eps, p, 1.0 / p])
    input_trt, eps_trt, p_trt, p_inv_trt = broadcast_trt_tensors(ctx.network, [input_trt, eps_trt, p_trt, p_inv_trt], len(input_trt.shape))
    
    # add tensorrt layer
    # compute norm = sum(abs(x)**p, dim=dim)**(1./p)
    norm = ctx.network.add_unary(input_trt, trt.UnaryOperation.ABS).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_reduce(norm, trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim, input.dim()), keep_dims=True).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_inv_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_elementwise(norm, eps_trt, trt.ElementWiseOperation.MAX).get_output(0)
    norm = ctx.network.add_elementwise(input_trt, norm, trt.ElementWiseOperation.DIV).get_output(0)
    
    # divide input by norm
    output._trt = norm