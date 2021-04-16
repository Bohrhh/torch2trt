from torch2trt.utils import *


def convert_cast(ctx, dtype):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    if isinstance(dtype, torch.dtype):
        trt_dtype = torch_dtype_to_trt(dtype)
    elif isinstance(dtype, torch.Tensor):
        trt_dtype = torch_dtype_to_trt(dtype.dtype)
    else:
        trt_dtype = torch_dtype_to_trt(input.dtype)
        
    layer = ctx.network.add_identity(input=input_trt)
    layer.set_output_type(0, trt_dtype)

    # get tensorrt output
    output._trt = layer.get_output(0)
    output._trt.shape # Supernatural phenomenon. Only after this line output._trt.dtype would be the dtype we set


@tensorrt_converter('torch.Tensor.type')
@tensorrt_converter('torch.Tensor.to')
def convert_cast_to_type(ctx):
    # parse args
    dtype  = get_arg(ctx, 'dtype', pos=1, default=None) 
    convert_cast(ctx, dtype)


@tensorrt_converter('torch.Tensor.bool')
def convert_cast_bool(ctx):
    convert_cast(ctx, torch.bool)


@tensorrt_converter('torch.Tensor.int')
def convert_cast_int(ctx):
    convert_cast(ctx, torch.int32)


@tensorrt_converter('torch.Tensor.long')
def convert_cast_long(ctx):
    convert_cast(ctx, torch.int64)


@tensorrt_converter('torch.Tensor.float')
def convert_cast_float(ctx):
    convert_cast(ctx, torch.float32)