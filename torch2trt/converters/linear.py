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
    assert sum([i==-1 for i in input_trt.shape])<=1, "Linear only support one dynamic dim"

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
    output_trt = layer.get_output(0)
    layer = ctx.network.add_shuffle(output_trt)
    layer.reshape_dims = output_trt.shape[:-2]

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

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)], dynamic_axes={0:[1,32]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)], dynamic_axes={1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], dynamic_axes={2:[4,40]})
def test_linear_dynamic():
    return torch.nn.Linear(10, 5)