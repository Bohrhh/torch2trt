from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.Tensor.permute')
def convert_permute(ctx):
    # parse args
    input  = ctx.method_args[0]
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c, d)
    else:
        permutation = tuple(ctx.method_args[1])   # handle permute([a, b, c, d])
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
   
    # get tensorrt output
    output._trt = layer.get_output(0)


class Permute(torch.nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args).contiguous()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0123():
    return TestInterface(lambda x: x.permute(0,1,2,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0312():
    return TestInterface(lambda x: x.permute(0,3,2,1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_3012():
    return TestInterface(lambda x: x.permute(3,0,2,1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_01234():
    return TestInterface(lambda x: x.permute(0,1,2,3,4))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_04132():
    return TestInterface(lambda x: x.permute(0,4,1,3,2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_list():
    return TestInterface(lambda x: x.permute([0,4,1,3,2]))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_tuple():
    return TestInterface(lambda x: x.permute((0,4,1,3,2)))