import torch
import numpy as np
import tensorrt as trt
from torch2trt.torch2trt import tensorrt_converter
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

    # add tensorrt layer
    creator = trt.get_plugin_registry().get_plugin_creator('GroupNormalizationPlugin', '1')
    assert creator is not None, 'Has no GroupNormalizationPlugin version 1'
    fc = trt.PluginFieldCollection()
    fc.append(trt.PluginField(name='eps',        data=np.array([eps],        dtype=np.float32), type=trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField(name='num_groups', data=np.array([num_groups], dtype=np.int32  ), type=trt.PluginFieldType.INT32  ))

    plugin = creator.create_plugin('group_num', fc)
    layer  = ctx.network.add_plugin_v2(inputs_trt, plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], enabled=trt_version() >= '7.1.3')
def test_group_norm_trt_g2_fp32():
    return torch.nn.GroupNorm(2, 10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], enabled=trt_version() >= '7.1.3')
def test_group_norm_trt_g2_eps_fp32():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112)], enabled=trt_version() >= '7.1.3')
def test_group_norm_trt_g2_eps_fp32():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)
