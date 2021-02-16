import torch.nn as nn
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None      ) 
    pad    = get_arg(ctx, 'pad',   pos=1, default=None      )
    mode   = get_arg(ctx, 'mode',  pos=2, default='constant')
    value  = get_arg(ctx, 'value', pos=3, default=0         )
    output = ctx.method_return
    assert mode=='constant' and value==0, "mode / value are ignored since not supported by TensorRT"
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    pre_padding = (pad[2], pad[0])
    post_padding = (pad[3], pad[1])
    layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)

    # get tensorrt output
    output._trt = layer.get_output(0)
        
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_pad_basic():
    return TestInterface(lambda x: nn.functional.pad(x, (1, 2, 3, 4)))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_pad_dynamic():
    return TestInterface(lambda x: nn.functional.pad(x, (1, 2, 3, 4)))