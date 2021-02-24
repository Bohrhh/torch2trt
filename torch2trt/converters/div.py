from torch2trt.utils import *


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