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


# dynamic_axes should be provided in this test example
# or the input binding can not be found
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_size_basic_dynamic():
    return TestInterface(lambda x: x.size())

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_size_dim2_dynamic():
    return TestInterface(lambda x: x.size(2))