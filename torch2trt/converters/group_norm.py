import torch
import numpy as np
import tensorrt as trt
from torch2trt.utils import *


@tensorrt_converter('torch.nn.GroupNorm.forward')
def convert_group_norm_trt(ctx):
    # parse args
    module     = ctx.method_args[0]
    input      = ctx.method_args[1]
    weight     = module.weight if module.weight is not None else torch.ones(module.num_channels, dtype=torch.float32)
    bias       = module.bias if module.bias is not None else torch.zeros(module.num_channels, dtype=torch.float32)
    eps        = module.eps
    num_groups = module.num_groups
    output     = ctx.method_return

    # get tensorrt input 
    inputs_trt = add_missing_trt_tensors(ctx.network, [input, weight, bias])
    assert not ctx.is_dynamic, "GroupNorm does not support dynamic shape now"
    

    # add tensorrt layer
    creator = trt.get_plugin_registry().get_plugin_creator('GroupNormalizationPlugin', '1')
    assert creator is not None, 'Has no GroupNormalizationPlugin version 1'
    fc = []
    fc.append(trt.PluginField(name='eps',        data=np.array([eps],        dtype=np.float32), type=trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField(name='num_groups', data=np.array([num_groups], dtype=np.int32  ), type=trt.PluginFieldType.INT32  ))
    fc = trt.PluginFieldCollection(fc)

    plugin = creator.create_plugin('group_num', fc)
    layer  = ctx.network.add_plugin_v2(inputs_trt, plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)
