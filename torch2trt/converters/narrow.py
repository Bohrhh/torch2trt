from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.narrow')
@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    # parse args
    input  = get_arg(ctx, 'input',  pos=0, default=None)  
    dim    = get_arg(ctx, 'dim',    pos=1, default=None)
    start  = get_arg(ctx, 'start',  pos=2, default=None)
    length = get_arg(ctx, 'length', pos=3, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    assert not ctx.is_dynamic, "Narrow does not support dynamic shape"

    # add tensorrt layer
    shape   = list(input.shape)
    starts  = [0]*input.dim()
    strides = [1]*input.dim()
    dim     = convert_dim(dim, input.dim())
    starts[dim] = start
    shape[dim]  = length 
    layer = ctx.network.add_slice(input=input_trt, start=starts, shape=shape, stride=strides)

    # get tensorrt output
    output._trt = layer.get_output(0)
