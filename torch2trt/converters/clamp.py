from torch2trt.utils import *


def __add_clamp(network, trt_input, val, op):
    
    # create TensorRT constant for minimum value
    val_shape  = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    val_tensor = val * torch.ones(val_shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
    layer      = network.add_constant(val_shape, val_tensor)
    layer      = network.add_elementwise(trt_input, layer.get_output(0), op)
    
    return layer

    
# CLAMP_MIN
@tensorrt_converter('torch.clamp_min')
@tensorrt_converter('torch.Tensor.clamp_min')
def convert_clamp_min(ctx):
    # parse args
    input  = ctx.method_args[0]
    val    = ctx.method_args[1]
    output = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MAX)
    
    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_min():
    return TestInterface(lambda x: torch.clamp_min(x, -0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_min():
    return TestInterface(lambda x: x.clamp_min(-0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_min_dynamic():
    return TestInterface(lambda x: x.clamp_min(-0.1))

    
# CLAMP_MAX
@tensorrt_converter('torch.clamp_max')
@tensorrt_converter('torch.Tensor.clamp_max')
def convert_clamp_max(ctx):
    # parse args
    input  = ctx.method_args[0]
    val    = ctx.method_args[1]
    output = ctx.method_return
    
    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MIN)
    
    # get tensorrt output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_max():
    return TestInterface(lambda x: torch.clamp_max(x, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_max():
    return TestInterface(lambda x: x.clamp_max(0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_max_dynamic():
    return TestInterface(lambda x: x.clamp_max(0.1))

# CLAMP 
@tensorrt_converter('torch.clamp')
@tensorrt_converter('torch.Tensor.clamp')
def convert_clamp(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None) 
    min    = get_arg(ctx, 'min',   pos=1, default=None) 
    max    = get_arg(ctx, 'max',   pos=2, default=None) 
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    if min is not None and max is not None:
        layer   = __add_clamp(ctx.network, input_trt, min, trt.ElementWiseOperation.MAX)
        layer   = __add_clamp(ctx.network, layer.get_output(0),  max, trt.ElementWiseOperation.MIN)
    elif min is not None:
        layer   = __add_clamp(ctx.network, input_trt, min, trt.ElementWiseOperation.MAX)
    elif max is not None:
        layer   = __add_clamp(ctx.network, input_trt, max, trt.ElementWiseOperation.MIN)
    else:
        assert False, "At least one of 'min' or 'max' must not be None!"
    
    # get tensorrt output
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp():
    return TestInterface(lambda x: torch.clamp(x, -0.1, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp():
    return TestInterface(lambda x: x.clamp(-0.1, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_max():
    return TestInterface(lambda x: torch.clamp(x, max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_min():
    return TestInterface(lambda x: torch.clamp(x, min=-0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_max_min():
    return TestInterface(lambda x: torch.clamp(x, min=-0.1, max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_option_max():
    return TestInterface(lambda x: x.clamp(max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_option_min():
    return TestInterface(lambda x: x.clamp(min=-0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_max_min():
    return TestInterface(lambda x: x.clamp(min=-0.1, max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_max_min_dynamic():
    return TestInterface(lambda x: x.clamp(min=-0.1, max=0.1))