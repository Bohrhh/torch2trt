from torch2trt.utils import *


@tensorrt_converter('torch.ones')
def convert_ones(ctx):
    # parse args
    if isinstance(ctx.method_args[0], tuple):
        shape = ctx.method_args[0]
    else:
        shape = ctx. method_args[:]
    output = ctx.method_return

    if not has_trt(*shape):
        return 

    # get tensorrt input 
    one = torch.tensor(1, dtype=output.dtype, device=output.device).reshape([1 for i in range(len(shape))])
    one_trt = add_missing_trt_tensors(ctx.network, [one])[0]
    sizes_trt = add_missing_trt_tensors(ctx.network, shape)

    # add tensorrt layer
    strides = tuple([0 for i in range(len(shape))])
    starts  = tuple([0 for i in range(len(shape))])
    sizes_trt = ctx.network.add_concatenation(sizes_trt).get_output(0)
    layer = ctx.network.add_slice(one_trt, starts, [], strides)
    layer.set_input(2, sizes_trt)

    output._trt = layer.get_output(0)
