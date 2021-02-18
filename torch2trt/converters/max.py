from torch2trt.utils import *


def __convert_max_elementwise(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MAX)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

def __convert_max_reduce(ctx):
    # parse args
    input      = get_arg(ctx, 'input',   pos=0, default=None )
    dim        = get_arg(ctx, 'dim',     pos=1, default=None )
    keepdim    = get_arg(ctx, 'keepdim', pos=2, default=False)
    if dim is not None:
        output_val = ctx.method_return[0]
        output_idx = ctx.method_return[1]
    else:
        output_val = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    if dim is None:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1)
        input_trt = layer.get_output(0)

    # add tensorrt layer
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MAX, 1, torch_dim_to_trt_axes(0 if dim is None else dim, input.dim()))

    if dim is None:
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_val.reshape_dims = (1, )
        output_val._trt = layer_val.get_output(0)
    elif keepdim:
        output_val._trt = layer.get_output(0)
        output_idx._trt = layer.get_output(1)
    else:
        assert sum([i==-1 for i in input_trt.shape])<=1, \
            "Max without keepdim only support one dynamic dim, please use keepdim for convenience"
        layer_val = ctx.network.add_shuffle(layer.get_output(0))
        layer_idx = ctx.network.add_shuffle(layer.get_output(1))
        shape = input_trt.shape[:dim] + input_trt.shape[dim+1:]
        layer_val.reshape_dims = tuple(shape)
        layer_idx.reshape_dims = tuple(shape)
        output_val._trt = layer_val.get_output(0)
        output_idx._trt = layer_idx.get_output(0)


@tensorrt_converter('torch.max')
@tensorrt_converter('torch.Tensor.max')
def convert_max(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_max_elementwise(ctx)
    else:
        __convert_max_reduce(ctx)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_dim1():
    return TestInterface(lambda x: torch.max(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_dim2():
    return TestInterface(lambda x: torch.max(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_dim1_keepdim():
    return TestInterface(lambda x: torch.max(x, 1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_all():
    return TestInterface(lambda x: torch.max(x))
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
def test_max_elementwise():
    return TestInterface(lambda x, y: torch.max(x,y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)], dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_max_d1_keepdim_dynamic():
    return TestInterface(lambda x: torch.max(x, 1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)], dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_max_reduce_all_dynamic():
    return TestInterface(lambda x: torch.max(x))
