from torch2trt.utils import *


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.matmul')
def convert_matmul(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    op_a = trt.MatrixOperation.VECTOR if input_a.dim()==1 else trt.MatrixOperation.NONE
    op_b = trt.MatrixOperation.VECTOR if input_b.dim()==1 else trt.MatrixOperation.NONE
    layer = ctx.network.add_matrix_multiply(input_a_trt, op_a, input_b_trt, op_b)
    
    # get tensorrt output
    output._trt = layer.get_output(0)
