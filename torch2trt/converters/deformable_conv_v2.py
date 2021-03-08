import torch
import numpy as np
import tensorrt as trt
from torch2trt.utils import *

@tensorrt_converter('torch2trt.plugins.ModulatedDeformConvFunction.forward', enabled=trt_version() >= '7.0')
def convert_dcnv2(ctx):
    # parse args
    print("=======dcnv2=========")
    input    = ctx.method_args[1]
    offset   = ctx.method_args[2]
    mask     = ctx.method_args[3]
    kernel   = ctx.method_args[4]
    bias     = ctx.method_args[5] if ctx.method_args[5] is not None else None
    stride   = get_arg(ctx, 'stride',   pos=6, default=1)
    padding  = get_arg(ctx, 'padding',  pos=7, default=0)
    dilation = get_arg(ctx, 'dilation', pos=8, default=1)
    groups   = get_arg(ctx, 'groups',   pos=9, default=1)
    deformable_groups = get_arg(ctx, 'deformable_groups', pos=10, default=1)
    output   = ctx.method_return

    assert input.dim()==4

    # get tensorrt input
    inputs_trt = add_missing_trt_tensors(ctx.network, [input, offset, mask, kernel])
    if bias is not None:
        bias_trt = add_missing_trt_tensors(ctx.network, [bias])[0]
        inputs_trt = inputs_trt + [bias_trt]
    
    # add tensorrt layer
    input_dim = input.dim() - 2
    if not isinstance(stride, tuple):
        stride      = (stride, ) * input_dim
    if not isinstance(padding, tuple):
        padding     = (padding, ) * input_dim
    if not isinstance(dilation, tuple):
        dilation    = (dilation, ) * input_dim

    creator = trt.get_plugin_registry().get_plugin_creator('ModulatedDeformableConvPlugin', '1')
    assert creator is not None, 'Has no ModulatedDeformableConvPlugin version 1'
    fc = []
    fc.append(trt.PluginField(name='stride',           data=np.array(stride,              dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='padding',          data=np.array(padding,             dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='dilation',         data=np.array(dilation,            dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='group',            data=np.array([groups],            dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='deformable_group', data=np.array([deformable_groups], dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc = trt.PluginFieldCollection(fc)

    plugin = creator.create_plugin('ModulatedDeformableConvPlugin', fc)
    layer  = ctx.network.add_plugin_v2(inputs_trt, plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)

