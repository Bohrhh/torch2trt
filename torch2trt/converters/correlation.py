import torch
import numpy as np
import tensorrt as trt
from torch2trt.utils import *


@tensorrt_converter('torch2trt.plugins.Correlation_TRT.forward')
def convert_group_norm_trt(ctx):
    # parse args
    module     = ctx.method_args[0]
    left       = ctx.method_args[1]
    right      = ctx.method_args[2]
    max_disp   = module.max_disp
    stride     = module.stride
    assert module.mode in ['l1', 'time']
    assert module.reduction in ['mean', 'sum']
    is_time    = 1 if module.mode=='time' else 0
    is_mean    = 1 if module.reduction=='mean' else 0
    output     = ctx.method_return

    # get tensorrt input 
    inputs_trt = add_missing_trt_tensors(ctx.network, [left, right])

    # add tensorrt layer
    creator = trt.get_plugin_registry().get_plugin_creator('CorrelationPlugin', '1')
    assert creator is not None, 'Has no CorrelationPlugin version 1'
    fc = []
    fc.append(trt.PluginField(name='max_disparity', data=np.array([max_disp], dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='stride',        data=np.array([stride],   dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='is_time',       data=np.array([is_time],  dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc.append(trt.PluginField(name='is_mean',       data=np.array([is_mean],  dtype=np.int32),   type=trt.PluginFieldType.INT32))
    fc = trt.PluginFieldCollection(fc)

    plugin = creator.create_plugin('CorrelationPlugin', fc)
    layer  = ctx.network.add_plugin_v2(inputs_trt, plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)
