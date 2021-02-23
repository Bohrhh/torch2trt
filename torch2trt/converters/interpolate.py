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


@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)], enabled=trt_version() >= '7.1')
def test_interpolate_nearest():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,5,13,13)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)], enabled=trt_version() >= '7.1')
def test_interpolate_bilinear():
    return torch.nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,12,12)], enabled=trt_version() >= '7.1')
def test_interpolate_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)], enabled=trt_version() >= '7.1')
def test_interpolate_size():
    return torch.nn.Upsample(size=3, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,13,13)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1)], enabled=trt_version() >= '7.1')
def test_interpolate_size_odd_input():
    return torch.nn.Upsample(size=[6,3], mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,6,6,6)], enabled=trt_version() >= '7.1')
def test_interpolate_nearest_3d():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,6,7,7,7)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,2,4,4)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1,1)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,5,5,5)], enabled=trt_version() >= '7.1')
def test_interpolate_bilinear_3d():
    return torch.nn.Upsample(scale_factor=3, mode="trilinear", align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,8,8,8)], enabled=trt_version() >= '7.1')
def test_interpolate_align_corner_3d():
    return torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)], enabled=trt_version() >= '7.1')
def test_interpolate_size_3d():
    return torch.nn.Upsample(size=3, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,7,9,5)], enabled=trt_version() >= '7.1')
@add_module_test(torch.float32, torch.device('cuda'), [(1,4,3,5,1)], enabled=trt_version() >= '7.1')
def test_interpolate_size_odd_input_3d():
    return torch.nn.Upsample(size=[11,14,17], mode="trilinear", align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)], enabled=trt_version() >= '7.1', dynamic_axes={0:[1,32], 2:[12,48], 3:[12,48]})
def test_interpolate_nearest_dynamic():
    return torch.nn.Upsample(size=[24,36], mode="nearest")
