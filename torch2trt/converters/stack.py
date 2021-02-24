from torch2trt.utils import *


@tensorrt_converter('torch.stack', enabled=trt_version() >= '7.0')
def convert_stack(ctx):
    # parse args
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    dim    = get_arg(ctx, 'dim',   pos=1, default=0   ) 
    output = ctx.method_return

    # get tensorrt inputs
    inputs_trt = add_missing_trt_tensors(ctx.network, inputs)

    # add tensorrt layer
    dim        = convert_dim(dim, inputs[0].dim())
    inputs_trt = [unsqueeze(ctx, i, dim) for i in inputs_trt]
    layer      = ctx.network.add_concatenation(inputs=inputs_trt)
    layer.axis = dim

    # get tensorrt output
    output._trt = layer.get_output(0)
