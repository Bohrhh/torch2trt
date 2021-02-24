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
