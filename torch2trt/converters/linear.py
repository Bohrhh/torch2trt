from torch2trt.torch2trt import tensorrt_converter
from torch2trt.utils import *


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    # parse args
    module    = ctx.method_args[0]
    input     = ctx.method_args[1]
    kernel    = module.weight.detach().cpu().numpy()
    bias      = module.bias.detach().cpu().numpy() if module.bias is not None else None
    output    = ctx.method_return

    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    # reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(input_trt.shape) + (1, 1) 
        
    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=module.out_features,
        kernel=kernel,
        bias=bias)

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape)

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_linear_basic():
    return torch.nn.Linear(10, 5)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)