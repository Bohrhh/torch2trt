import torch
import numpy as np
import tensorrt as trt
from torch2trt.utils import *


@tensorrt_converter('torch.gather')
def convert_grid_sample(ctx):
    # parse args
    input         = ctx.method_args[0]
    dim           = ctx.method_args[1]
    index         = ctx.method_args[2]
    output        = ctx.method_return

    dim = convert_dim(dim, input.dim())

    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    index_trt = add_missing_trt_tensors(ctx.network, [index])[0]

    # add tensorrt layer
    creator = trt.get_plugin_registry().get_plugin_creator('GatherElementsPlugins', '1')
    assert creator is not None, 'Has no GatherElementsPlugins version 1'
    fc = []
    fc.append(trt.PluginField(name='dim', data=np.array([dim], dtype=np.int32), type=trt.PluginFieldType.INT32))
    fc = trt.PluginFieldCollection(fc)

    plugin = creator.create_plugin('GatherElementsPlugins', fc)
    layer  = ctx.network.add_plugin_v2([input_trt, index_trt], plugin)

    # get tensorrt output
    output._trt = layer.get_output(0)
