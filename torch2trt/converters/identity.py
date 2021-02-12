import torch.nn as nn
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *

@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_functional_identity(ctx):
    input       = ctx.method_args[0]
    input_trt   = add_missing_trt_tensors(ctx.network, [input])[0]
    output      = ctx.method_return
    output._trt = input_trt


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_identity(ctx):
    input       = ctx.method_args[1]
    input_trt   = add_missing_trt_tensors(ctx.network, [input])[0]
    output      = ctx.method_return
    output._trt = input_trt


@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_tensor_contiguous():
    return TestInterface(lambda x: (x+1).contiguous())

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_dropout_f():
    return TestInterface(lambda x: nn.functional.dropout((x+1), training=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_dropout2d_f():
    return TestInterface(lambda x: nn.functional.dropout2d((x+1), training=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3,3)])
def test_dropout3d_f():
    return TestInterface(lambda x: nn.functional.dropout3d((x+1), training=False))

