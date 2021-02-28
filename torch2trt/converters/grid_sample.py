import torch
import numpy as np
import tensorrt as trt
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.grid_sample')
def convert_group_norm_trt(ctx):
    # parse args
    input         = ctx.method_args[0]
    grid          = ctx.method_args[1]
    mode          = get_arg(ctx, 'mode',          pos=2, default='bilinear')
    padding_mode  = get_arg(ctx, 'padding_mode',  pos=3, default='zeros')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)
    output     = ctx.method_return

    mode = ['bilinear', 'nearest'].index(mode)
    padding_mode = ['zeros', 'border', 'reflection'].index(padding_mode)
    align_corners = 1 if align_corners else 0

    # get tensorrt input 
    inputs_trt = add_missing_trt_tensors(ctx.network, [input, grid])

    # add tensorrt layer
    creator = trt.get_plugin_registry().get_plugin_creator('GridSamplePlugin', '1')
    assert creator is not None, 'Has no GridSamplePlugin version 1'
    fc = []
    fc.append(trt.PluginField(name='mode',          data=np.array([mode],           dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='padding_mode',  data=np.array([padding_mode],   dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='align_corners', data=np.array([align_corners],  dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc = trt.PluginFieldCollection(fc)

    plugin = creator.create_plugin('GridSamplePlugin', fc)
    layer  = ctx.network.add_plugin_v2(inputs_trt, plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)
