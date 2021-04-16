from torch2trt.utils import *
logger = get_root_logger()

def convert_elementwise(ctx, trt_op, dtype=None):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b], dtype=dtype)
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())
    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt_op)

    # get tensorrt output
    output._trt = layer.get_output(0)


def convert_relementwise(ctx, trt_op, dtype=None):
    # parse args
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output  = ctx.method_return

    # get tensorrt inputs
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b], dtype=dtype)
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
    convert_elementwise(ctx, trt.ElementWiseOperation.SUM, dtype=ctx.method_return.dtype)


# ============================================================
# sub

@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.SUB, dtype=ctx.method_return.dtype)


@tensorrt_converter('torch.Tensor.__rsub__')
def convert_rsub(ctx):
    convert_relementwise(ctx, trt.ElementWiseOperation.SUB, dtype=ctx.method_return.dtype)


# ============================================================
# mul

@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.PROD, dtype=ctx.method_return.dtype)


# ============================================================
# div

@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.DIV, dtype=ctx.method_return.dtype)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    convert_relementwise(ctx, trt.ElementWiseOperation.DIV, dtype=ctx.method_return.dtype)


@tensorrt_converter('torch.floor_divide')
@tensorrt_converter('torch.Tensor.__floordiv__')
@tensorrt_converter('torch.Tensor.__ifloordiv__')
def convert_floordiv(ctx):
    logger.error('operation "//" only positive result would give correct result')
    convert_elementwise(ctx, trt.ElementWiseOperation.FLOOR_DIV, dtype=ctx.method_return.dtype)


# ============================================================
# &

@tensorrt_converter('torch.Tensor.__and__')
def convert_and(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.AND)


# ============================================================
# |

@tensorrt_converter('torch.Tensor.__or__')
def convert_or(ctx):
    convert_elementwise(ctx, trt.ElementWiseOperation.OR)


# ============================================================
# ^

@tensorrt_converter('torch.Tensor.__xor__')
def convert_xor(ctx):
    convert_elementwise(ctx, trt_op=trt.ElementWiseOperation.XOR)


# ============================================================
# >

@tensorrt_converter('torch.gt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__gt__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.GREATER)


# ============================================================
# <

@tensorrt_converter('torch.lt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__lt__', enabled=trt_version() >= '7.0')
def convert_lt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.LESS)


# ============================================================
# ==

@tensorrt_converter('torch.eq', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__eq__', enabled=trt_version() >= '7.0')
def convert_eq(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.EQUAL)


# ============================================================
# <=
@tensorrt_converter('torch.le', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__le__', enabled=trt_version() >= '7.0')
def convert_le(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.GREATER)
    layer = ctx.network.add_unary(layer.get_output(0), trt.UnaryOperation.NOT)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# >=
@tensorrt_converter('torch.ge', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__ge__', enabled=trt_version() >= '7.0')
def convert_ge(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.LESS)
    layer = ctx.network.add_unary(layer.get_output(0), trt.UnaryOperation.NOT)

    # get tensorrt output
    output._trt = layer.get_output(0)