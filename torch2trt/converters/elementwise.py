from torch2trt.utils import *


def convert_elementwise(ctx, trt_op):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt_op)

    # get tensorrt output
    output._trt = layer.get_output(0)


def convert_relementwise(ctx, trt_op):
    # parse args
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output  = ctx.method_return

    # get tensorrt inputs
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt_op)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# add

@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
@tensorrt_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.SUM)


# ============================================================
# sub

@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.SUB)


@tensorrt_converter('torch.Tensor.__rsub__')
def convert_rsub(ctx):
    convert_relementwise(ctx, trt.ElementWiseOperation.SUB)


# ============================================================
# mul

@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.PROD)


# ============================================================
# div

@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.DIV)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    convert_relementwise(ctx, trt.ElementWiseOperation.DIV)


# ============================================================
# and

@tensorrt_converter('torch.Tensor.__and__')
def convert_and(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.AND)


# ============================================================
# or

@tensorrt_converter('torch.Tensor.__or__')
def convert_or(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.OR)


# ============================================================
# or

@tensorrt_converter('torch.Tensor.__or__')
def convert_xor(ctx):
    convert_elementwise(ctx, trt_op=trt.ElementWiseOperation.XOR)


# ============================================================
# greater

@tensorrt_converter('torch.gt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__gt__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.GREATER)


# ============================================================
# less

@tensorrt_converter('torch.lt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__lt__', enabled=trt_version() >= '7.0')
def convert_lt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.LESS)


# ============================================================
# equal

@tensorrt_converter('torch.eq', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__eq__', enabled=trt_version() >= '7.0')
def convert_eq(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.EQUAL)