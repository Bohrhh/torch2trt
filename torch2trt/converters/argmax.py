from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
    

@tensorrt_converter('torch.argmax')
@tensorrt_converter('torch.Tensor.argmax')
def convert_argmax(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MAX, 1, torch_dim_to_trt_axes(dim))
    output._trt = layer.get_output(1)

        
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_argmax_dim1():
    return UnaryModule(lambda x: torch.argmax(x, dim=1, keepdim=True))
