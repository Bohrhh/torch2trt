from torch2trt.utils import *


def slice_to_trt(dim_size, dim_slice):
    
    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step
    
    size = (stop - start - 1) // stride + 1
    
    return start, size, stride


def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int):
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
    assert not ctx.is_dynamic, "Getitem do not support dynamic shape"

    # add tensorrt layer
    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = input.dim() - num_slice_types(slices)
    
    new_slices = []
    for s in slices:
        
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
        elif s is None:
            new_slices.append(None)
        elif isinstance(s, int):
            new_slices.append(s)
            
    # fill missing slices at end
    while num_slice_types(new_slices) < input.dim():
        new_slices.append(slice(None, None, None))
            
    slices = tuple(new_slices)
    
    # Step 2 - Add slice layer (will currently ignore 'None' slices)
    starts = []
    sizes = []
    strides = []
    
    input_dim = 0
    for s in slices:
        
        if input_dim >= len(input_trt.shape):
            break
            
        input_size = int(input.shape[input_dim])
        
        if isinstance(s, slice):
            start, size, stride = slice_to_trt(input_size, s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            input_dim += 1
            
        elif isinstance(s, int):
            starts.append(s)
            sizes.append(1)
            strides.append(1)
            input_dim += 1
    
    output_trt = ctx.network.add_slice(input_trt, starts, sizes, strides).get_output(0)
    
    # Step 3 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices
    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:
        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output.shape) # exclude batch
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