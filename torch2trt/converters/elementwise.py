from torch2trt.utils import *


# ============================================================
# add

@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
@tensorrt_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# sub

@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt inputs
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)

    # get tensorrt  output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rsub__')
def convert_rsub(ctx):
    # parse args
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output  = ctx.method_return

    # get tensorrt inputs
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# mul

@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# div

@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    # parse args
    input_a = ctx.method_args[1]  # inputs switched for rdiv
    input_b = ctx.method_args[0]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ============================================================
# and

@tensorrt_converter('torch.Tensor.__and__')
def convert_and(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.AND)

    # get tensorrt output
    output._trt = layer.get_output(0)



# ============================================================
# or

@tensorrt_converter('torch.Tensor.__or__')
def convert_or(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.OR)

    # get tensorrt output
    output._trt = layer.get_output(0)