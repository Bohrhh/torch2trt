from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.narrow')
@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    # parse args
    input  = get_arg(ctx, 'input',  pos=0, default=None)  
    dim    = get_arg(ctx, 'dim',    pos=1, default=None)
    start  = get_arg(ctx, 'start',  pos=2, default=None)
    length = get_arg(ctx, 'length', pos=3, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    assert all([i!=-1 for i in input_trt.shape]), "Narrow does not support dynamic shape"

    # add tensorrt layer
    shape   = list(input.shape)
    starts  = [0]*input.dim()
    strides = [1]*input.dim()
    dim     = convert_dim(dim, input.dim())
    starts[dim] = start
    shape[dim]  = length 
    layer = ctx.network.add_slice(input=input_trt, start=starts, shape=shape, stride=strides)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)])
def test_narrow1():
    return TestInterface(lambda x: torch.narrow(x, 1, 0, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)])
def test_narrow2():
    return TestInterface(lambda x: torch.narrow(x, 2, 2, 50))
