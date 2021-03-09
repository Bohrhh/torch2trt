from torch2trt.utils import *



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
