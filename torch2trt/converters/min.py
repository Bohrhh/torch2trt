from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


def __convert_min_elementwise(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MIN)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

def __convert_min_reduce(ctx):
    # parse args
    input      = get_arg(ctx, 'input',   pos=0, default=None )
    dim        = get_arg(ctx, 'dim',     pos=1, default=None )
    keepdim    = get_arg(ctx, 'keepdim', pos=2, default=False)
    output_val = ctx.method_return[0]
    if dim is not None:
        output_idx = ctx.method_return[1]

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    if dim is None:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1)
        input_trt = layer.get_output(0)

    # add tensorrt layer
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MIN, 1, torch_dim_to_trt_axes(0 if dim is None else dim, input.dim()))

    if dim is None:
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_val.reshape_dims = (1, )
        output_val._trt = layer_val.get_output(0)
    elif keepdim:
        output_val._trt = layer.get_output(0)
        output_idx._trt = layer.get_output(1)
    else:
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_idx = ctx.network.add_shuffle(layer.get_output(1))
        shape = input.shape[:dim] + input.shape[dim+1:]
        layer_val.reshape_dims = tuple(shape)
        layer_idx.reshape_dims = tuple(shape)
        output_val._trt = layer.get_output(0)
        output_idx._trt = layer.get_output(1)


@tensorrt_converter('torch.min')
@tensorrt_converter('torch.Tensor.min')
def convert_min(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_min_elementwise(ctx)
    else:
        __convert_min_reduce(ctx)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_val():
    return TestInterface(lambda x: torch.min(x, 1)[0])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_idx():
    return TestInterface(lambda x: torch.min(x, 1)[1])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim2_val():
    return TestInterface(lambda x: torch.min(x, 2)[0])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim2_idx():
    return TestInterface(lambda x: torch.min(x, 2)[1])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_keepdim_val():
    return TestInterface(lambda x: torch.min(x, 1, keepdim=True)[0])    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_keepdim_idx():
    return TestInterface(lambda x: torch.min(x, 1, keepdim=True)[1])  

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_all():
    return TestInterface(lambda x: torch.min(x))
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
def test_min_elementwise():
    return TestInterface(lambda x, y: torch.min(x,y))
