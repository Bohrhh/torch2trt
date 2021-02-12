from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_basic():
    return TestInterface(lambda x, y: x*y)

class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_imul():
    return IMul()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_torchmul():
    return TestInterface(lambda x, y: torch.mul(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_int():
    return TestInterface(lambda x: 10*x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_float():
    return TestInterface(lambda x: 10.0*x)

class MulConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x * self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_nobatch():
    return MulConstantNoBatch()

class MulConstantBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x * self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_batch():
    return MulConstantBatch()
