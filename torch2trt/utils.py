import torch
import importlib
import tensorrt as trt


# UTILITY FUNCTIONS


def trt_version():
    return trt.__version__


def torch_dtype_to_trt(dtype):
    if trt_version() >= '7.0' and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.int64:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def trt_num_inputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            count += 1
    return count


def trt_num_outputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            count += 1
    return count


def convert_dim(dim, length):
    """Converts negative dim
    Args:
        dim: int or tuple
        length: length of input shape
    """
    if isinstance(dim, (tuple, list)):
        dim = tuple([length+d if d<0 else d for d in dim])
    else:
        dim = length+dim if dim<0 else dim
    return dim


def torch_dim_to_trt_axes(dim, length):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask
    Args:
        dim: int or tuple
        length: length of input shape
    """
    dim = convert_dim(dim, length)
    if not isinstance(dim, tuple):
        dim = (dim,)
    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


def add_trt_constant(network, tensor):
    shape = tuple(tensor.shape[1:])
    array = tensor[0].detach().cpu().numpy()
    layer = network.add_constant(shape, array)
    return layer.get_output(0)


def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'Tensor data types must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype

    
def add_missing_trt_tensors(network, tensors):
    """Creates missing TensorRT tensors as constants and attaches them to the Torch Tensors"""
    trt_tensors = [None] * len(tensors)

    dtype = check_torch_dtype(*tensors)

    for i, t in enumerate(tensors):
        trt_tensor = None

        # GET TRT TENSOR (OR CREATE TRT CONSTANT)

        # get tensor w/ _trt
        # or... add constant for scalar primitive
        if isinstance(t, float) or isinstance(t, int):
            shape = (1,)
            scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
            trt_tensor = network.add_constant(shape, scalar).get_output(0)
        elif hasattr(t, "_trt"):
            trt_tensor = t._trt

        # or... add constant for leaf tensor w/o _trt
        else:
            
            # remove all preceding ones, these can be re-inserted later when broadcasting
            num_preceding_ones = 0
            for j in range(len(t.shape)):
                if int(t.shape[j]) == 1:
                    num_preceding_ones += 1
                else:
                    break
            shape = tuple(t.shape[num_preceding_ones:])
            
            weight = t.detach().cpu().numpy()
            t._trt = network.add_constant(shape, weight).get_output(0)
            trt_tensor = t._trt


        assert trt_tensor is not None

        trt_tensors[i] = trt_tensor

    return trt_tensors
    

def broadcast_trt_tensors(network, trt_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    broadcasted_trt_tensors = [None] * len(trt_tensors)
    
    for i, t in enumerate(trt_tensors):
        
        if len(t.shape) < broadcast_ndim:
            # append 1 size dims to front
            diff = broadcast_ndim - len(t.shape)
            shape = tuple([1] * diff + list(t.shape))
            layer = network.add_shuffle(t)
            layer.reshape_dims = shape
            trt_tensor = layer.get_output(0)
        else:
            trt_tensor = t

        broadcasted_trt_tensors[i] = trt_tensor
        
    return broadcasted_trt_tensors
    
    
def trt_(network, *tensors):
    """Creates missing TensorRT tensors and adds shuffle layers to make tensors broadcastable"""
    trt_tensors = [None] * len(tensors)

    dtype = check_torch_dtype(*tensors)

    # get broadcast dimension
    broadcast_num_dim = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if not hasattr(t, "_trt"):
                num_dim = len(t.shape)  # don't exclude batch for constants
            else:
                num_dim = len(
                    t._trt.shape
                )  # non-leaf tensors must already have _trt, get shape from that
            if num_dim > broadcast_num_dim:
                broadcast_num_dim = num_dim

    for i, t in enumerate(tensors):
        trt_tensor = None

        # GET TRT TENSOR (OR CREATE TRT CONSTANT)

        # get tensor w/ _trt
        if isinstance(t, torch.Tensor) and hasattr(t, "_trt"):
            trt_tensor = t._trt

        # or... add constant for leaf tensor w/o _trt
        elif isinstance(t, torch.Tensor) and not hasattr(t, "_trt"):
            # add leaf tensor
            shape = tuple(t.shape)  #  don't exclude batch when adding constants...?
            weight = t.detach().cpu().numpy()
            t._trt = network.add_constant(shape, weight).get_output(0)
            trt_tensor = t._trt

        # or... add constant for scalar primitive
        elif isinstance(t, float) or isinstance(t, int):
            shape = (1,) * broadcast_num_dim
            scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
            trt_tensor = network.add_constant(shape, scalar).get_output(0)

        assert trt_tensor is not None

        # MAKE TRT TENSOR BROADCASTABLE IF IT IS NOT ALREADY

        if len(trt_tensor.shape) < broadcast_num_dim:
            # append 1 size dims to front
            diff = broadcast_num_dim - len(trt_tensor.shape)
            shape = tuple([1] * diff + list(trt_tensor.shape))
            layer = network.add_shuffle(trt_tensor)
            layer.reshape_dims = shape
            trt_tensor = layer.get_output(0)

        trt_tensors[i] = trt_tensor

    if len(trt_tensors) == 1:
        return trt_tensors[0]
    else:
        return tuple(trt_tensors)


def get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default


def attach_converter(ctx, method, converter, method_str):
    """Gets a function that executes PyTorch method and TensorRT converter"""
    global DUMMY_CONVERTERS

    def wrapper(*args, **kwargs):
        skip = True

        # check if another (parent) converter has lock
        if not ctx.lock:
            if converter["is_real"]:
                ctx.lock = True  # only real converters can acquire lock
            skip = False

        # run original method
        outputs = method(*args, **kwargs)

        if not skip:
            ctx.method_args = args
            ctx.method_kwargs = kwargs
            ctx.method_return = outputs
            ctx.method_str = method_str

            #             print('%s' % (converter.__name__,))
            converter["converter"](ctx)

            # convert to None so conversion will fail for unsupported layers
            ctx.method_args = None
            ctx.method_kwargs = None
            ctx.method_return = None
            ctx.lock = False

        return outputs

    return wrapper


# DEFINE ALL CONVERSION FUNCTIONS
def get_module_qualname(name):
    s = name.split('.')
    
    for i in range(len(s)):
        idx = len(s) - i - 1
        modulename, qualname = ".".join(s[:idx]), ".".join(s[idx:])
        try:
            module = importlib.import_module(modulename)
            return module, modulename, qualname
        except:
            pass
        
    raise RuntimeError("Could not import module")
    

# TEST MODULE
class TestInterface(torch.nn.Module):
    def __init__(self, fn):
        super(TestInterface, self).__init__()
        self.fn = fn
        
    def forward(self, *x):
        return self.fn(*x)

class ModuleTest(object):
    def __init__(self, module_fn, dtype, device, input_shapes, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def module_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__


MODULE_TESTS = []


def add_module_test(dtype, device, input_shapes, enabled=True, **torch2trt_kwargs):
    def register_module_test(module):
        global MODULE_TESTS
        MODULE_TESTS += [ModuleTest(module, dtype, device, input_shapes, **torch2trt_kwargs)]
        return module

    def pass_module_test(module):
        return module

    if enabled:
        return register_module_test
    else:
        return pass_module_test

    return register_module_test