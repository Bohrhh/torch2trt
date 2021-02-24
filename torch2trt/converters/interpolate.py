import torch
import collections
import torch.nn as nn  
import torch.nn.functional as F    
from torch2trt.utils import *

                                                  
@tensorrt_converter('torch.nn.functional.interpolate', enabled=trt_version() >= '7.1')
@tensorrt_converter('torch.nn.functional.upsample', enabled=trt_version() >= '7.1')
def convert_interpolate(ctx):                                     
    # parse args                     
    input         = get_arg(ctx, 'input',         pos=0, default=None     ) 
    size          = get_arg(ctx, 'size',          pos=1, default=None     )
    scale_factor  = get_arg(ctx, 'scale_factor',  pos=2, default=None     )
    mode          = get_arg(ctx, 'mode',          pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None     )
    output        = ctx.method_return

    # get tensorrt input 
    input_trt   = add_missing_trt_tensors(ctx.network, [input])[0]
    input_shape = ctx.network.add_shape(input=input_trt).get_output(0)

    # add tensorrt layer
    input_dim    = input.dim() - 2

    # size
    if size is not None:
        if not isinstance(size, collections.Sequence):
            size = [size] * input_dim
        size_trts  = add_missing_trt_tensors(ctx.network, size)
        layer      = ctx.network.add_concatenation(inputs=size_trts)
        layer.axis = 0
        size_trt   = layer.get_output(0)
        input_NC = ctx.network.add_slice(input=input_shape, start=[0], shape=[2], stride=[1]).get_output(0)
        layer = ctx.network.add_concatenation(inputs=[input_NC, size_trt])
        layer.axis = 0
        output_shape = layer.get_output(0)
    
    if scale_factor is not None:
        if not isinstance(scale_factor, collections.Sequence):
            scale_factor = [scale_factor] * input_dim
        scale_factor = [1,1] + scale_factor
        scale_factor = torch.tensor(scale_factor, device=input.device, dtype=torch.int32)
        scale_factor_trt = add_missing_trt_tensors(ctx.network, [scale_factor])[0]
        output_shape = ctx.network.add_elementwise(scale_factor_trt, input_shape, trt.ElementWiseOperation.PROD).get_output(0)
    
    layer = ctx.network.add_resize(input=input_trt)
    layer.set_input(1, output_shape)

    # other
    resize_mode = mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.NEAREST

    if align_corners != None:
        layer.align_corners = align_corners

    output._trt = layer.get_output(0)
