from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.transpose', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.transpose', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.transpose_', enabled=trt_version() >= '7.0')
def convert_transpose(ctx):
    # parse args
    input  = ctx.method_args[0]
    dim0   = ctx.method_args[1]
    dim1   = ctx.method_args[2]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    permutation = list(range(input.dim()))
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)])
def test_torch_transpose_02():
    return TestInterface(lambda x: x.transpose(1, 2))

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)])
def test_tensor_transpose_02():
    return TestInterface(lambda x: x.transpose(0, 2))

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)])
def test_tensor_transpose_12():
    return TestInterface(lambda x: x.transpose(1, 2))