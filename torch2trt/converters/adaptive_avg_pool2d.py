import numpy as np
from torch2trt.utils import *


@tensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
def convert_adaptive_avg_pool2d(ctx):
    # parse args
    input       = ctx.method_args[0]
    output_size = ctx.method_args[1]
    output      = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    if isinstance(output_size, int):
        output_size = (output_size, ) * 2
    
    
    if all([i==1 for i in output_size]):
        # deal with global avg pool
        reduce_axes = torch_dim_to_trt_axes(tuple(range(2, input.dim())), input.dim())
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, reduce_axes, True)
    else:
        assert all([i!=-1 for i in input_trt.shape[1:]]), "Input has dynamic shape except batch"

        kernel_size = (int(np.ceil(float(input_trt.shape[2])/output_size[0])), 
                       int(np.ceil(float(input_trt.shape[3])/output_size[1])))
        
        stride = ((input_trt.shape[2]-kernel_size[0])//(output_size[0]-1) if output_size[0]>1 else 1,
                  (input_trt.shape[3]-kernel_size[1])//(output_size[1]-1) if output_size[1]>1 else 1)
        
        assert stride[0]*(output_size[0]-1)+kernel_size[0]==input_trt.shape[2], \
            "Input width:{}, output width:{} would make trt kernel size or stride inconsistent.".format(input_trt.shape[2], output_size[0])
        assert stride[1]*(output_size[1]-1)+kernel_size[1]==input_trt.shape[3], \
            "Input height:{}, output height:{} would make trt kernel size or stride inconsistent.".format(input_trt.shape[2], output_size[0])

        layer = ctx.network.add_pooling(input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
        layer.stride = stride

    # get tensorrt output
    output._trt = layer.get_output(0)


