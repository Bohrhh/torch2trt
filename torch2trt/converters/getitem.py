import torch
from collections.abc import Iterable
from torch2trt.utils import *


def slice_to_trt(dim_size, dim_slice):
    start = 0 if dim_slice.start is None else convert_dim(dim_slice.start, dim_size)
    stop = dim_size if dim_slice.stop is None else convert_dim(dim_slice.stop, dim_size)
    stride = 1 if dim_slice.step is None else dim_slice.step
    size = (stop - start - 1) // stride + 1
    return start, size, stride


def slice_to_trt_dynamic(ctx, dim_size_trt, dim_slice):
    start = 0 if dim_slice.start is None else dim_slice.start
    start_trt = add_missing_trt_tensors(ctx.network, [start], dtype=torch.int32)[0]
    if start < 0:
        start_trt = ctx.network.add_elementwise(dim_size_trt, start_trt, trt.ElementWiseOperation.SUM).get_output(0)

    if dim_slice.stop is None:
        stop_trt = dim_size_trt
    else:
        stop_trt = add_missing_trt_tensors(ctx.network, [dim_slice.stop], dtype=torch.int32)[0]
        if dim_slice.stop < 0:
            stop_trt = ctx.network.add_elementwise(dim_size_trt, stop_trt, trt.ElementWiseOperation.SUM).get_output(0)
    stride = 1 if dim_slice.step is None else dim_slice.step
    stride_trt = add_missing_trt_tensors(ctx.network, [stride], dtype=torch.int32)[0]
    one_trt = add_missing_trt_tensors(ctx.network, [1], dtype=torch.int32)[0]
    size_trt = ctx.network.add_elementwise(stop_trt, start_trt, trt.ElementWiseOperation.SUB).get_output(0)
    size_trt = ctx.network.add_elementwise(size_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
    size_trt = ctx.network.add_elementwise(size_trt, stride_trt, trt.ElementWiseOperation.DIV).get_output(0)
    size_trt = ctx.network.add_elementwise(size_trt, one_trt, trt.ElementWiseOperation.SUM).get_output(0)
    return start_trt, size_trt, stride_trt



def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int) or isinstance(s, Iterable):
            num_slice += 1
    return num_slice


def convert_getitem_slice(ctx):
    # parse args
    input  = ctx.method_args[0]
    slices = ctx.method_args[1]
    if not isinstance(slices, tuple):
        slices = (slices, )
    output = ctx.method_return

    # get tensorrt input 
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = input.dim() - num_slice_types(slices)
    
    new_slices = []
    new_gathers = []

    for s in slices:
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                new_gathers.append(None)
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
            new_gathers.append(None)
        elif s is None:
            new_slices.append(None)
            new_gathers.append(None)
        elif isinstance(s, int):
            new_slices.append(s)
            new_gathers.append(None)
        elif isinstance(s, Iterable):
            new_slices.append(slice(None, None, None))
            new_gathers.append(s)
            
    # fill missing slices at end
    while num_slice_types(new_slices) < input.dim():
        new_slices.append(slice(None, None, None))
        new_gathers.append(None)

    slices = tuple(new_slices)
    
    # Step 2 - Add slice layer (will currently ignore 'None' slices)
    starts  = []
    sizes   = []
    strides = []

    if ctx.is_dynamic:
        starts_trt  = []
        sizes_trt   = []
        strides_trt = []
        shape_trt = ctx.network.add_shape(input_trt).get_output(0)

    input_dim = 0
    for s in slices:
        
        if input_dim >= len(input_trt.shape):
            break
        if ctx.is_dynamic:
            input_size_trt = ctx.network.add_slice(input=shape_trt, start=[input_dim], shape=[1], stride=[1]).get_output(0)
            
        if isinstance(s, slice):
            input_size = int(input.shape[input_dim])
            start, size, stride = slice_to_trt(input_size, s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            
            if ctx.is_dynamic:
                start_trt, size_trt, stride_trt = slice_to_trt_dynamic(ctx, input_size_trt, s)
                starts_trt.append(start_trt)
                sizes_trt.append(size_trt)
                strides_trt.append(stride_trt)

            input_dim += 1

        elif isinstance(s, int):
            input_size = int(input.shape[input_dim])
            starts.append(convert_dim(s, input_size))
            sizes.append(1)
            strides.append(1)

            if ctx.is_dynamic:
                one_trt = add_missing_trt_tensors(ctx.network, [1], dtype=torch.int32)[0]
                start_trt = add_missing_trt_tensors(ctx.network, [s], dtype=torch.int32)[0]
                if s<0:
                    start_trt = ctx.network.add_elementwise(input_size_trt, start_trt, trt.ElementWiseOperation.SUM).get_output(0)
                starts_trt.append(start_trt)
                sizes_trt.append(one_trt)
                strides_trt.append(one_trt)

            input_dim += 1
    
    if ctx.is_dynamic:
        starts_trt = ctx.network.add_concatenation(starts_trt).get_output(0)
        sizes_trt = ctx.network.add_concatenation(sizes_trt).get_output(0)
        strides_trt = ctx.network.add_concatenation(strides_trt).get_output(0)
        layer = ctx.network.add_slice(input_trt, [], [], [])
        layer.set_input(1, starts_trt)
        layer.set_input(2, sizes_trt)
        layer.set_input(3, strides_trt)
        output_trt = layer.get_output(0)
    else:
        output_trt = ctx.network.add_slice(input_trt, starts, sizes, strides).get_output(0)
    
    # Step 3 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices
    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:
        if ctx.is_dynamic:
            n_remove = 0
            for i, s in enumerate(slices):
                if s is None:
                    output_trt = unsqueeze(ctx, output_trt, i-n_remove)
                if isinstance(s, int):
                    output_trt = squeeze(ctx, output_trt, i-n_remove)
                    n_remove += 1
        else:
            layer = ctx.network.add_shuffle(output_trt)
            layer.reshape_dims = tuple(output.shape)
            output_trt = layer.get_output(0)
        
    output._trt = output_trt
        

@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx):
    # parse args
    x     = ctx.method_args[1]

    if isinstance(x, torch.Tensor):
        assert False, 'Do not implemented tensor idx now'
    elif isinstance(x, tuple) and any([isinstance(i, list) for i in x]):
        assert False, 'Do not implemented list idx now'
    elif isinstance(x, list):
        assert False, 'Do not implemented list idx now'
    else:
        convert_getitem_slice(ctx)