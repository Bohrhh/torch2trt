from torch2trt.utils import *


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
    