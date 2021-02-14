from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx):
    # parse args
    input   = get_arg(ctx, 'input',   pos=0, default=None )
    dim     = get_arg(ctx, 'dim',     pos=1, default=tuple(range(len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    output  = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, torch_dim_to_trt_axes(dim, input.dim()), keepdim)

    # get tensorrt output
    output._trt = layer.get_output(0)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_torch_d1():
    return TestInterface(lambda x: torch.mean(x, dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tensor_d1():
    return TestInterface(lambda x: x.mean(dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tensor_reduce():
    return TestInterface(lambda x: x.mean())

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_torch_d1_d2():
    return TestInterface(lambda x: torch.mean(x, dim=(1,2), keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tensor_d1_d2():
    return TestInterface(lambda x: x.mean(dim=[1, 2], keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_torch_keepdim():
    return TestInterface(lambda x: torch.mean(x, dim=1, keepdim=True))