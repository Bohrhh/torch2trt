from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *

@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
@tensorrt_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output  = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return TestInterface(lambda x, y: x+y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_dynamic():
    return TestInterface(lambda x, y: x+y)

class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_iadd():
    return IAdd()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_iadd_dynamic():
    return IAdd()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torch():
    return TestInterface(lambda x, y: torch.add(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_int():
    return TestInterface(lambda x: 1 + x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_float():
    return TestInterface(lambda x: 1.0 + x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_radd_float_dynamic():
    return TestInterface(lambda x: 1.0 + x)

class AddConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x + self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_add_constant_nobatch():
    return AddConstantNoBatch()

class AddConstantBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x + self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_add_constant_batch():
    return AddConstantBatch()
