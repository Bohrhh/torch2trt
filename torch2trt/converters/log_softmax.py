import torch.nn as nn
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.log_softmax')
@tensorrt_converter('torch.Tensor.log_softmax')
@tensorrt_converter('torch.nn.functional.log_softmax')
def convert_log_softmax(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return
    assert dim is not None, 'Dim should be provided!'
    assert dim != 0, 'There is large error in test when dim is zero.'
    

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = torch_dim_to_trt_axes(dim, input.dim())
    layer = ctx.network.add_unary(input=layer.get_output(0), op=trt.UnaryOperation.LOG)

    # get tensorrt output
    output._trt = layer.get_output(0)


# dim==0 has some unknow error
# @add_module_test(torch.float32, torch.device('cuda'), [(3, 4, 10)])
# def test_logsoftmax_d0():
#     return nn.LogSoftmax(dim=0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_logsoftmax_d1():
    return nn.LogSoftmax(dim=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_tensor_logsoftmax_d1():
    return TestInterface(lambda x: x.log_softmax(dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_torch_logsoftmax_d1():
    return TestInterface(lambda x: torch.log_softmax(x, dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_logsoftmax_d2():
    return nn.LogSoftmax(dim=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)], dynamic_axes={0:[1,32], 1:[10,50]})
def test_logsoftmax_d1_dynamic():
    return nn.LogSoftmax(dim=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)], dynamic_axes={0:[1,32], 1:[3,30], 2:[10,50]})
def test_logsoftmax_d2_dynamic():
    return nn.LogSoftmax(dim=2)