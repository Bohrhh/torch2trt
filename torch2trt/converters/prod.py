from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *
 

@tensorrt_converter('torch.prod')
@tensorrt_converter('torch.Tensor.prod')
def convert_prod(ctx):
    # parse args
    input   = get_arg(ctx, 'input',   pos=0, default=None)
    dim     = get_arg(ctx, 'dim',     pos=1, default=tuple(range(input.dim())))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    output  = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.PROD, torch_dim_to_trt_axes(dim, input.dim()), keepdim)

    # get tensorrt output
    output._trt = layer.get_output(0)

        
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_all():
    return TestInterface(lambda x: torch.prod(x))     

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim1():
    return TestInterface(lambda x: torch.prod(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim2():
    return TestInterface(lambda x: torch.prod(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim1_keepdim():
    return TestInterface(lambda x: torch.prod(x, 1, keepdim=True))
