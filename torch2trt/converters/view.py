from functools import reduce
from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *

@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.unsqueeze')
def convert_view(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(output.shape)

    # get tensorrt output
    output._trt = layer.get_output(0)

@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.flatten')
def convert_flatten(ctx):
    # parse args
    input     = get_arg(ctx, 'input'    , pos=0, default=None)
    start_dim = get_arg(ctx, 'start_dim', pos=1, default=0   )
    end_dim   = get_arg(ctx, 'end_dim'  , pos=2, default=-1  )
    output    = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]
    end_dim    = input.dim()-1 if end_dim==-1 else end_dim
    flatten    = input_trt.shape[start_dim:end_dim+1]
    shape_pre  = input_trt.shape[:start_dim]
    shape_post = input_trt.shape[end_dim+1:]
    if -1 in flatten:
        flatten = -1
    else:
        flatten = reduce(lambda x,y:x*y, flatten)
    shape = shape_pre+(flatten, )+shape_post
    assert sum([i==-1 for i in shape])<=1, "trt shuffle operation only support one -1 shape"    

    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = shape

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_flatten_1d():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)])
def test_flatten_batch2_1d():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], dynamic_axes={0:[1,32]})
def test_flatten_1d_dynamic():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], dynamic_axes={1:[3,30], 2:[3,30]})
def test_flatten_1d_dynamic2():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return TestInterface(lambda x: x.view(1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)])
def test_view_batch2_1d():
    return TestInterface(lambda x: x.view(2, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return TestInterface(lambda x: x.view(1, 1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)])
def test_view_3d():
    return TestInterface(lambda x: x.view(1, 3, 3, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)])
def test_unsqueeze():
    return TestInterface(lambda x: x.unsqueeze(dim=2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)])
def test_squeeze():
    return TestInterface(lambda x: x.squeeze(dim=2))


