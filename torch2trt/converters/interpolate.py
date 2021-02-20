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
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    assert all([i!=-1 for i in input_trt.shape]), "Interpolate does not support dynamic shape now"
    
    # add tensorrt layer
    input_dim = input.dim() - 2
    layer = ctx.network.add_resize(input=input_trt)

    shape = size
    if shape != None:
        if isinstance(shape, collections.Sequence):
            shape  = list(input_trt.shape[:2]) + list(shape)
        else:
            shape = list(input_trt.shape[:2]) + [shape] * input_dim

        layer.shape = shape

    scales = scale_factor
    if scales != None:
        if not isinstance(scales, collections.Sequence):
            scales = [scales] * input_dim
        shape = list(input_trt.shape[:2]) + [input_trt.shape[i+2]*int(scales[i]) for i in range(input_dim)]
        # layer.scales = [1] + list(scales)
        layer.shape = shape

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