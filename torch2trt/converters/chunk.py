from torch2trt.torch2trt import tensorrt_converter
from .split import convert_split
from torch2trt.utils import *


@tensorrt_converter('torch.chunk')
@tensorrt_converter('torch.Tensor.chunk')
def convert_chunk(ctx):
    convert_split(ctx)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_c1d1():
    return TestInterface(lambda x: torch.chunk(x, 1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_c2d1():
    return TestInterface(lambda x: torch.chunk(x, 2, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_c3d1():
    return TestInterface(lambda x: torch.chunk(x, 3, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_c3d2():
    return TestInterface(lambda x: torch.chunk(x, 3, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tensor_chunk_c3d2():
    return TestInterface(lambda x: x.chunk(3, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], dynamic_axes={0:[1,32], 2:[3, 30], 3:[3, 30]})
def test_tensor_chunk_c2d1_dynamic():
    return TestInterface(lambda x: x.chunk(2, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], dynamic_axes={1:[1,32]})
def test_tensor_chunk_c2d1_dynamic1():
    return TestInterface(lambda x: x.chunk(2, 1))