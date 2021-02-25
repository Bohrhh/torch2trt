from functools import reduce
from torch2trt.utils import *


@tensorrt_converter('torch.reshape')
@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
def convert_view(ctx):
    # parse args
    input  = ctx.method_args[0]
    if isinstance(ctx.method_args[1], int):
        shape = tuple(ctx.method_args[1:])  # handle permute(a, b, c, d)
    else:
        shape = tuple(ctx.method_args[1])   # handle permute([a, b, c, d])
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    shape_trt = add_missing_trt_tensors(ctx.network, shape, dtype=torch.int32)

    # add tensorrt layer
    layer = ctx.network.add_concatenation(inputs=shape_trt)
    layer.axis = 0
    output_shape_trt = layer.get_output(0)
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, output_shape_trt)

    # get tensorrt output
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.squeeze_')
def convert_squeeze(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None)
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]

    if dim is None:
        reduce_dim = 0
        for i, s in enumerate(input.shape):
            if s==1:
                input_trt = squeeze(ctx, input_trt, i-reduce_dim)
                reduce_dim += 1
    else:
        input_trt = squeeze(ctx, input_trt, dim)

    # get tensorrt output
    output._trt = input_trt


@tensorrt_converter('torch.unsqueeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.Tensor.unsqueeze_')
def convert_unsqueeze(ctx):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None)
    dim    = get_arg(ctx, 'dim',   pos=1, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]

    # get tensorrt output
    output._trt = unsqueeze(ctx, input_trt, dim)


@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.flatten')
def convert_flatten(ctx):
    # parse args
    input     = get_arg(ctx, 'input'    , pos=0, default=None)
    start_dim = get_arg(ctx, 'start_dim', pos=1, default=0   )
    end_dim   = get_arg(ctx, 'end_dim'  , pos=2, default=-1  )
    output    = ctx.method_return

    # get tensorrt input
    input_trt  = add_missing_trt_tensors(ctx.network, [input])[0]
    start_dim  = convert_dim(start_dim, input.dim())
    end_dim    = convert_dim(end_dim,   input.dim())
    flatten    = input_trt.shape[start_dim:end_dim+1]

    if start_dim == end_dim:
        output._trt = input_trt
    else:
        if -1 in flatten:
            flatten = -1
        else:
            flatten = reduce(lambda x,y:x*y, flatten)

        flatten_trt    = add_missing_trt_tensors(ctx.network, [flatten], dtype=torch.int32)[0]
        pre_shape_trt  = None
        post_shape_trt = None

        input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
        if start_dim!=0:
            pre_shape_trt  = ctx.network.add_slice(input=input_shape_trt, start=[0],       shape=[start_dim],             stride=[1]).get_output(0)
        if end_dim!=input.dim()-1:
            post_shape_trt = ctx.network.add_slice(input=input_shape_trt, start=[end_dim], shape=[input.dim()-end_dim-1], stride=[1]).get_output(0)

        if pre_shape_trt is None and post_shape_trt is None:
            output_shape_trt = flatten_trt
        elif pre_shape_trt is None:
            layer = ctx.network.add_concatenation(inputs=[flatten_trt, post_shape_trt])
            layer.axis = 0
            output_shape_trt = layer.get_output(0)
        elif post_shape_trt is None:
            layer = ctx.network.add_concatenation(inputs=[pre_shape_trt, flatten_trt])
            layer.axis = 0
            output_shape_trt = layer.get_output(0)
        else:
            layer = ctx.network.add_concatenation(inputs=[pre_shape_trt, flatten_trt, post_shape_trt])
            layer.axis = 0
            output_shape_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, output_shape_trt)
        output._trt = layer.get_output(0)