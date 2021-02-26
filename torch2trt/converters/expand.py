from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    # parse args
    input  = ctx.method_args[0]
    sizes  = ctx.method_args[1:]
    output = ctx.method_return

    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    assert not ctx.is_dynamic, "Expand do not support dynamic shape"

    # add tensorrt layer
    inshape  = tuple(input.shape) # exclude batch
    outshape = tuple([i if i!=-1 else j for i,j in zip(sizes, input_trt.shape)])
    ndim     = len(outshape)
    start    = tuple([0]*ndim)
    stride   = tuple([int(i == o) for i, o in zip(inshape, outshape)])  # stride == 1 if dimensions match, 0 otherwise
    layer    = ctx.network.add_slice(input_trt, start, outshape, stride)
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    