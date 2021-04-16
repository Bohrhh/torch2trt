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
