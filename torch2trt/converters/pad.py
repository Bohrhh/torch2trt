import torch.nn as nn
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None      ) 
    pad    = get_arg(ctx, 'pad',   pos=1, default=None      )
    mode   = get_arg(ctx, 'mode',  pos=2, default='constant')
    value  = get_arg(ctx, 'value', pos=3, default=0         )
    output = ctx.method_return
    assert input.dim()==4 or input.dim()==3, '3 or 4 dimensions are required for input'
    assert (mode=='constant' or mode=='reflect') and value==0, 'mode:{} / value:{} are ignored since not supported by TensorRT'.format(mode, value)
    assert len(pad)<=4, 'Only 2D padding is currently supported.'

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    if (input.dim()==3 or input.dim()==4) and (mode=='constant' and value==0) and len(pad)<=4:
        # using tensorrt original operation

        if input.dim()==3:
            input_trt = unsqueeze(ctx, input_trt, -2)
        
        if len(pad)==2:
            pad = pad+(0,0)

        # add tensorrt layer
        pre_padding = (pad[2], pad[0])
        post_padding = (pad[3], pad[1])
        layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)

        output_trt = layer.get_output(0)

        if input.dim()==3:
            output_trt = squeeze(ctx, output_trt, -2)

        # get tensorrt output
        output._trt = output_trt
    elif (input.dim()==3 or input.dim()==4) and mode=='reflect' and len(pad)<=4:
        # using tensrrt plugins
        pad = np.array(list(pad)+[1 for i in range(16-len(pad))]).astype(np.int32)

        # add tensorrt layer
        creator = trt.get_plugin_registry().get_plugin_creator('PaddingPlugin', '1')
        assert creator is not None, 'Has no PaddingPlugin version 1'
        fc = []
        fc.append(trt.PluginField(name='padding', data=np.array(pad, dtype=np.int32), type=trt.PluginFieldType.INT32))
        fc.append(trt.PluginField(name='mode',    data=np.array([2], dtype=np.int32), type=trt.PluginFieldType.INT32))
        fc = trt.PluginFieldCollection(fc)

        plugin = creator.create_plugin('PaddingPlugin', fc)
        layer  = ctx.network.add_plugin_v2([input_trt], plugin)

        # get tensorrt output
        output._trt = layer.get_output(0)
