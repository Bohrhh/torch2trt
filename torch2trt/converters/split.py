from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.split')
@tensorrt_converter('torch.Tensor.split')
def convert_split(ctx):
    # parse args
    input   = get_arg(ctx, 'input', pos=0, default=None)
    dim     = get_arg(ctx, 'dim',   pos=2, default=0   )
    outputs = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    start  = [0] * input.dim() # exclude batch
    stride = [1] * len(start)
    offset = 0
    dim    = convert_dim(dim, input.dim())
    
    # add slice layers
    for i, output in enumerate(outputs):
        shape       = list(output.shape) # exclude batch dim
        start[dim]  = offset
        layer       = ctx.network.add_slice(input_trt, start=start, shape=shape, stride=stride)
        output._trt = layer.get_output(0)
        offset      = offset + shape[dim]
        

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)])
def test_torch_split_1_d0():
    return TestInterface(lambda x: torch.split(x, 1, 0))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_1_d1():
    return TestInterface(lambda x: torch.split(x, 1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_2_d1():
    return TestInterface(lambda x: torch.split(x, 2, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_3_d1():
    return TestInterface(lambda x: torch.split(x, 3, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_1_d2():
    return TestInterface(lambda x: x.split(1, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tensor_split_2_d2():
    return TestInterface(lambda x: x.split(2, 2))