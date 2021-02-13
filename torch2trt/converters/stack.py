from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


def unsqueeze(ctx, input, dim):
    layer = ctx.network.add_shuffle(input)
    shape = input.shape[:dim] + (1,) + input.shape[dim:]
    layer.reshape_dims = tuple(shape)
    return layer.get_output(0)


@tensorrt_converter('torch.stack', enabled=trt_version() >= '7.0')
def convert_cat(ctx):
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


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0')
def test_Stack_dim1():
    return TestInterface(lambda *x: torch.stack(x, dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0')
def test_Stack_dim3():
    return TestInterface(lambda *x: torch.stack(x, dim=3))