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
    assert input.dim()==4, 'At least 4 dimensions are required for input'
    assert mode=='constant' and value==0, 'mode:{} / value:{} are ignored since not supported by TensorRT'.format(mode, value)
    assert len(pad)<=4, 'Only 2D padding is currently supported.'

    
    if input.dim()==4 and (mode=='constant' and value==0) and len(pad)<=4:
        # using tensorrt original operation
        if len(pad)==2:
            pad = pad+(0,0)
        # get tensorrt input
        input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
        
        # add tensorrt layer
        pre_padding = (pad[2], pad[0])
        post_padding = (pad[3], pad[1])
        layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)

        # get tensorrt output
        output._trt = layer.get_output(0)
    else:
        # using tensrrt plugins
        pass
