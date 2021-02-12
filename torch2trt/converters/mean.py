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

    
class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        return x.mean(self.dim, self.keepdim)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_channel():
    return Mean(1, False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tuple():
    return Mean((1, 2), False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_keepdim():
    return Mean(1, True)