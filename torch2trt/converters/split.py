from torch2trt.utils import *


@tensorrt_converter('torch.split')
@tensorrt_converter('torch.Tensor.split')
def convert_split(ctx):
    # parse args
    input   = get_arg(ctx, 'input', pos=0, default=None)
    dim     = get_arg(ctx, 'dim',   pos=2, default=0   )
    outputs = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    assert not ctx.is_dynamic, "Split do not support dynamic shape"
    
    start  = [0] * input.dim()
    stride = [1] * len(start)
    offset = 0
    dim    = convert_dim(dim, input.dim())
    
    # add slice layers
    for i, output in enumerate(outputs):
        shape       = list(output.shape)
        start[dim]  = offset
        layer       = ctx.network.add_slice(input_trt, start=start, shape=shape, stride=stride)
        output._trt = layer.get_output(0)
        offset      = offset + shape[dim]


