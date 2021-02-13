from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.unsqueeze')
def convert_view(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(output.shape[1:])

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return TestInterface(lambda x: x.view(1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return TestInterface(lambda x: x.view(1, 1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)])
def test_view_3d():
    return TestInterface(lambda x: x.view(1, 3, 3, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)])
def test_unsqueeze():
    return TestInterface(lambda x: x.unsqueeze(dim=2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)])
def test_squeeze():
    return TestInterface(lambda x: x.squeeze(dim=2))


