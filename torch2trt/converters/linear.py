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
    input_trt = unsqueeze(ctx, input_trt, -1)
    input_trt = unsqueeze(ctx, input_trt, -1)
        
    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=input_trt,
        num_outputs=module.out_features,
        kernel=kernel,
        bias=bias)

    # reshape back to N
    output_trt = layer.get_output(0)
    output_trt = squeeze(ctx, output_trt, -1)
    output_trt = squeeze(ctx, output_trt, -1)

    output._trt = output_trt


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
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)], dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], dynamic_axes={0:[1,32], 1:[3,30], 2:[4,40]})
def test_linear_dynamic():
    return torch.nn.Linear(10, 5)