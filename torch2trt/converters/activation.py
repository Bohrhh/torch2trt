from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


#  |    RELU         : Relu activation: f(x) = x if x >= 0, f(x) = 0 if x < 0
#  |    RELU6        : Relu6 activation: f(x) = x if 0 <= x <= 6, f(x) = 0 if x < 0, f(x) = 6 if x > 6
#  |    SIGMOID      : Sigmoid activation
#  |    TANH         : Hyperbolic Tangent activation
#  |    LEAKY_RELU   : Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0
#  |    ELU          : Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0
#  |    SELU         : Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0
#  |    SOFTSIGN     : Softsign activation: f(x) = x / (1 + \|x\|)
#  |    SOFTPLUS     : Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)
#  |    HARD_SIGMOID : Hard sigmoid activation: f(x) = max(0, min(1, alpha * x + beta))


# ========================================================================
# Relu activation: f(x) = x if x >= 0, f(x) = 0 if x < 0
@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
def convert_relu(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input=input_trt, type=trt.ActivationType.RELU)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu_basic():
    return torch.nn.ReLU()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu_basic():
    return TestInterface(lambda x: torch.nn.functional.relu(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_relu_dynamic():
    return torch.nn.ReLU()


# ========================================================================
# Relu6 activation: f(x) = x if 0 <= x <= 6, f(x) = 0 if x < 0, f(x) = 6 if x > 6
@tensorrt_converter('torch.nn.functional.relu6')
def convert_functional_relu6(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input, 6])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], output.dim())

    layer = ctx.network.add_activation(input=input_a_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_relu6(ctx):
    ctx.method_args = ctx.method_args[1:]
    convert_functional_relu6(ctx)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu6_basic():
    return torch.nn.ReLU6()
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu6_basic():
    return TestInterface(lambda x: torch.nn.functional.relu6(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_relu6_dynamic():
    return torch.nn.ReLU6()


# ========================================================================
# Sigmoid activation
@tensorrt_converter('torch.nn.functional.sigmoid')
@tensorrt_converter('torch.sigmoid')
def convert_sigmoid(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_sigmoid_basic():
    return torch.nn.Sigmoid()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_sigmoid_dynamic():
    return torch.nn.Sigmoid()

# ========================================================================
# Tanh
@tensorrt_converter('torch.nn.functional.tanh')
@tensorrt_converter('torch.tanh')
def convert_tanh(ctx):
    # parse args
    input  = ctx.method_args[0]
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    # add tensorrt output
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.TANH)

    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_tanh_basic():
    return torch.nn.Tanh()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_tanh_dynamic():
    return torch.nn.Tanh()


# ========================================================================
# Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0
@tensorrt_converter('torch.nn.functional.leaky_relu')
@tensorrt_converter('torch.nn.functional.leaky_relu_')
def convert_leaky_relu(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    negative_slope = get_arg(ctx, 'negative_slope', pos=1, default=0.01)
    output = ctx.method_return
    
    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_leaky_relu():
    return TestInterface(lambda x: torch.nn.functional.leaky_relu(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_leaky_relu_dynamic():
    return TestInterface(lambda x: torch.nn.functional.leaky_relu(x))


# ========================================================================
# Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0
@tensorrt_converter('torch.nn.functional.elu')
@tensorrt_converter('torch.nn.functional.elu_')
def convert_elu(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    alpha = get_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return
    
    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.ELU)
    layer.alpha = alpha
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_elu():
    return TestInterface(lambda x: torch.nn.functional.elu(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_elu_dynamic():
    return TestInterface(lambda x: torch.nn.functional.elu(x))


# ========================================================================
# Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0
@tensorrt_converter('torch.selu')
@tensorrt_converter('torch.selu_')
@tensorrt_converter('torch.nn.functional.selu')
@tensorrt_converter('torch.nn.functional.selu_')
def convert_selu(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SELU)
    layer.alpha = 1.6732632423543772848170429916717
    layer.beta = 1.0507009873554804934193349852946
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_selu():
    return TestInterface(lambda x: torch.nn.functional.selu(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_selu_dynamic():
    return TestInterface(lambda x: torch.nn.functional.selu(x))


# ========================================================================
# Softsign activation: f(x) = x / (1 + abs(x))
@tensorrt_converter('torch.nn.functional.softsign')
def convert_softsign(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTSIGN)
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_softsign():
    return TestInterface(lambda x: torch.nn.functional.softsign(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_softsign_dynamic():
    return TestInterface(lambda x: torch.nn.functional.softsign(x))


# ========================================================================
# Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)
@tensorrt_converter('torch.nn.functional.softplus')
def convert_softplus(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTPLUS)
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_softplus():
    return TestInterface(lambda x: torch.nn.functional.softplus(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_softplus_dynamic():
    return TestInterface(lambda x: torch.nn.functional.softplus(x))


# ========================================================================
# Hard sigmoid activation: f(x) = max(0, min(1, 1/6 * x + 0.5))
@tensorrt_converter('torch.nn.functional.hardsigmoid')
def convert_hardsigmoid(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.HARD_SIGMOID)
    layer.alpha = 1/6
    layer.beta = 0.5
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)])
def test_hardsigmoid():
    return TestInterface(lambda x: torch.nn.functional.hardsigmoid(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], dynamic_axes={0:[1,32], 2:[4,40]})
def test_hardsigmoid_dynamic():
    return TestInterface(lambda x: torch.nn.functional.hardsigmoid(x))