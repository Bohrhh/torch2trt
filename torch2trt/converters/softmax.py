from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return
    assert dim is not None, 'Dim should be provided!'
    # assert dim != 0, 'There is large error in test when dim is zero.'

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = torch_dim_to_trt_axes(dim, input.dim())

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(4, 3, 3, 3)])
def test_softmax_module_dim0():
    return torch.nn.Softmax(0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim1():
    return torch.nn.Softmax(1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim2():
    return torch.nn.Softmax(2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim_neg1():
    return torch.nn.Softmax(-1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim_neg2():
    return torch.nn.Softmax(-2)
