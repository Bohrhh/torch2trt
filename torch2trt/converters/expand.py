from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    # parse args
    input  = ctx.method_args[0]
    sizes  = ctx.method_args[1:]
    output = ctx.method_return

    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    inshape = tuple(input.shape) # exclude batch
    shape   = tuple(output.shape)
    ndim    = len(shape)
    start   = tuple([0]*ndim)
    stride  = tuple([int(i == o) for i, o in zip(inshape, shape)])  # stride == 1 if dimensions match, 0 otherwise
    layer = ctx.network.add_slice(input_trt, start, shape, stride)
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_tensor_expand_singledim():
    return TestInterface(lambda x: x.expand(1,3,3,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)])
def test_tensor_expand_multidim():
    return TestInterface(lambda x: x.expand(1,3,3,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)])
def test_tensor_expand_inferdim():
    return TestInterface(lambda x: x.expand(1,3,-1,-1))