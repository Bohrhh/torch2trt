from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    # parse args
    input   = get_arg(ctx, 'input', pos=0, default=None)
    dim     = get_arg(ctx, 'dim',   pos=1, default=None)
    outputs = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    shape_trt = ctx.network.add_shape(input=input_trt).get_output(0)
    if dim is not None:
        indices_trt = add_missing_trt_tensors(ctx.network, [dim])[0]
        outputs._trt = ctx.network.add_gather(shape_trt, indices_trt, axis=0).get_output(0)
    else:
        for i in range(input.dim()):
            indices_trt = add_missing_trt_tensors(ctx.network, [i])[0]
            outputs[i]._trt = ctx.network.add_gather(shape_trt, indices_trt, axis=0).get_output(0)
