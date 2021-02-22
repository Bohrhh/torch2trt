import copy
import torch
import importlib
import numpy as np
import tensorrt as trt

# GLOBAL VARIABLE
CONVERTERS = {}
MODULE_TESTS = []

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


def torch_dtype_to_numpy(dtype):
    if dtype == torch.bool:
        return np.bool
    elif dtype == torch.int8:
        return np.int8
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int32
    elif dtype == torch.float16:
        return np.float16
    elif dtype == torch.float32:
        return np.float32
    else:
        raise TypeError("%s is not supported by numpy" % dtype)


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


def to_tuple(x):
    if isinstance(x, list):
        x = tuple(x)
    if isinstance(x, torch.Size) or not isinstance(x, tuple):
        x = (x, )
    return x


def default_input_names(num_inputs):
    return ["input_%d" % i for i in range(num_inputs)]


def default_output_names(num_outputs):
    return ["output_%d" % i for i in range(num_outputs)]


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

    for i, t in enumerate(tensors):
        trt_tensor = None

        # GET TRT TENSOR (OR CREATE TRT CONSTANT)

        # get tensor w/ _trt
        # or... add constant for scalar primitive
        if isinstance(t, (float, int)):
            shape = (1,)
            scalar = torch.tensor([t])
            scalar = scalar.detach().cpu().numpy().astype(torch_dtype_to_numpy(scalar.dtype))
            print("=============")
            print(scalar)
            print(scalar.dtype)
            print(type(scalar))
            print(scalar.shape)
            layer = network.add_constant(shape, scalar)
            print(layer.shape)
            trt_tensor = layer.get_output(0)
            print(trt_tensor.shape)

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
            
            weight = t.detach().cpu().numpy().astype(torch_dtype_to_numpy(t.dtype))
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


def get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default


def convert_shape_tensor(tensor):
    if hasattr(tensor, 'is_shape_tensor'):
        if tensor.is_shape_tensor and tensor.dim()==0:
            return tensor.item()
        elif tensor.is_shape_tensor and tensor.dim()>0:
            return tuple(tensor.to('cpu').numpy())
        else:
            return tensor
    else:
        return tensor


def convert_input(x):
    if isinstance(x, torch.Tensor):
        return convert_shape_tensor(x)
    elif isinstance(x, list):
        return [convert_input(i) for i in x]
    elif isinstance(x, tuple):
        return tuple([convert_input(i) for i in x])


def extract_torch_inputs(args, kwargs):
    torch_args = []
    torch_kwargs = {}
    for a in args:
        torch_args.append(convert_input(a))
    for k,v in kwargs.items():
        torch_kwargs[k] = convert_input(v)
    return torch_args, torch_kwargs


def has_shape_tensor(inputs):
    if isinstance(inputs, (tuple, list)):
        for i in inputs:
            if isinstance(i, torch.Tensor) and hasattr(i, 'is_shape_tensor') and i.is_shape_tensor:
                return True
            elif isinstance(i, (tuple, list, dict)):
                return has_shape_tensor(i)
            else:
                return False
    if isinstance(inputs, dict):
        for k,v in inputs.items():
            if isinstance(v, torch.Tensor) and hasattr(v, 'is_shape_tensor') and v.is_shape_tensor:
                return True
            elif isinstance(v, (tuple, list, dict)):
                return has_shape_tensor(v)
            else:
                return False


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
        try:
            outputs = method(*args, **kwargs)

            if not skip:
                # special for torch.Tensor.size method
                if method_str=="torch.Tensor.size":
                    outputs = torch.tensor(outputs, device=args[0].device)
                    outputs.is_shape_tensor = True

                if has_shape_tensor(args) or has_shape_tensor(kwargs):
                    if isinstance(outputs, (tuple, list)):
                        for o in outputs:
                            o.is_shape_tensor = True
                    else:
                        outputs.is_shape_tensor = True

        except TypeError:
            torch_args, torch_kwargs = extract_torch_inputs(args, kwargs)
            outputs = method(*torch_args, **torch_kwargs)


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


def tensorrt_converter(method, is_real=True, enabled=True, imports=[]):
    
    if isinstance(method, str):
        module, module_name, qual_name = get_module_qualname(method)
    else:
        module, module_name, qual_name = importlib.import_module(method.__module__), method.__module__, method.__qualname__
        
    method_impl = eval('copy.deepcopy(module.%s)' % qual_name)
    
    def register_converter(converter):
        CONVERTERS[method] = {
            "converter": converter, 
            "is_real": is_real, 
            "module": module,
            "module_name": module_name,
            "qual_name": qual_name,
            "method_str": module_name + '.' + qual_name,
            "method_impl": method_impl
        }
        return converter

    def pass_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return pass_converter

    return register_converter