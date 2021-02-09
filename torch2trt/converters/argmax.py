from torch2trt.torch2trt import tensorrt_converter
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
from torch2trt.utils import *
    
    
@tensorrt_converter('torch.argmax')
@tensorrt_converter('torch.Tensor.argmax')
def convert_argmax(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    if dim is None:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1)
        input_trt = layer.get_output(0)
    
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MAX, 1, torch_dim_to_trt_axes(0 if dim is None else dim))

    if dim is None:
        layer = ctx.network.add_shuffle(layer.get_output(1))
        layer.reshape_dims = (1, )
        output._trt = layer.get_output(0)
    elif keepdim:
        output._trt = layer.get_output(1)
    else:
        layer = ctx.network.add_shuffle(layer.get_output(1))
        shape = input.shape[:dim] + input.shape[dim+1:]
        layer.reshape_dims = tuple(shape)
        output._trt = layer.get_output(0)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_argmax_dim1():
    return UnaryModule(lambda x: torch.argmax(x, dim=1, keepdim=False))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_argmax_dim2():
    return UnaryModule(lambda x: torch.argmax(x, dim=2, keepdim=False))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_argmax_dim3():
    return UnaryModule(lambda x: torch.argmax(x, dim=3, keepdim=False))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_argmax_keepdim():
    return UnaryModule(lambda x: torch.argmax(x, dim=1, keepdim=True))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_argmax_tensor():
    return UnaryModule(lambda x: x.argmax(dim=1, keepdim=True))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_argmax_tensor():
    return UnaryModule(lambda x: x.argmax())