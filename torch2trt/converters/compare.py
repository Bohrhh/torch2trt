from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


def convert_elementwise(ctx, op):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, op)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.gt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__gt__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.GREATER)


@tensorrt_converter('torch.lt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__lt__', enabled=trt_version() >= '7.0')
def convert_lt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.LESS)


@tensorrt_converter('torch.eq', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__eq__', enabled=trt_version() >= '7.0')
def convert_eq(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.EQUAL)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_gt_tensor():
    return TestInterface(lambda x, y: x>y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_gt_float():
    return TestInterface(lambda x: x>0.1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_lt_tensor():
    return TestInterface(lambda x, y: x<y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_lt_float():
    return TestInterface(lambda x: x<0.1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_eq_tensor():
    return TestInterface(lambda x, y: x==y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_eq_float():
    return TestInterface(lambda x: x==0.1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0', dynamic_axes={0:[1,32], 2:[6,60], 3:[6,60]})
def test_gt_tensor_dynamic():
    return TestInterface(lambda x, y: x>y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0', dynamic_axes={0:[1,32], 2:[6,60], 3:[6,60]})
def test_gt_float_dynamic():
    return TestInterface(lambda x: x>0.1)