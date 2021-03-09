from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.type')
@tensorrt_converter('torch.Tensor.to')
def convert_cast(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    dtype  = get_arg(ctx, 'dtype', pos=1, default=None) 
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    trt_type = torch_dtype_to_trt(dtype) 
    layer = ctx.network.add_identity(input=input_trt)
    layer.set_output_type(0, trt_type)

    # get tensorrt output
    output._trt = layer.get_output(0)
