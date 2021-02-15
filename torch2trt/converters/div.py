from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    # parse args
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    # parse args
    input_a = ctx.method_args[1]  # inputs switched for rdiv
    input_b = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    # add tensorrt layer
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)

    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_basic():
    return TestInterface(lambda x, y: x / y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[100,400], 3:[100,400]})
def test_div_dynamic():
    return TestInterface(lambda x, y: x / y)

class IDiv(torch.nn.Module):
    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_idiv():
    return IDiv()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[100,400], 3:[100,400]})
def test_div_idiv_dynamic():
    return IDiv()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_torch():
    return TestInterface(lambda x, y: torch.div(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rdiv_int():
    return TestInterface(lambda x: 10 / x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rdiv_float():
    return TestInterface(lambda x: 10.0 / x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[100,400], 3:[100,400]})
def test_rdiv_float_dynamic():
    return TestInterface(lambda x: 10.0 / x)

class DivConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(DivConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x / self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_div_constant_nobatch():
    return DivConstantNoBatch()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)], dynamic_axes={0:[1,32], 2:[10,40], 3:[10,40]})
def test_div_constant_nobatch_dynamic():
    return DivConstantNoBatch()

class DivConstantBatch(torch.nn.Module):
    def __init__(self):
        super(DivConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x / self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_div_constant_batch():
    return DivConstantBatch()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)], dynamic_axes={0:[1,32], 2:[10,40], 3:[10,40]})
def test_div_constant_batch_dynamic():
    return DivConstantBatch()