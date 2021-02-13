import copy
import torch
import importlib
import tensorrt as trt
from collections import defaultdict
from .utils import *

from .calibration import (
    TensorBatchDataset,
    DatasetCalibrator,
    DEFAULT_CALIBRATION_ALGORITHM,
)


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
    

# CONVERSION REGISTRY AND HOOKS

CONVERTERS = {}


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, key, converter):
        self.ctx = ctx
        self.key = key
        self.converter = converter

    def _set_method(self, method):
        module = self.converter['module']
        exec('module.%s = method' % self.converter['qual_name'])

    def __enter__(self):
        self._set_method(
            attach_converter(
                self.ctx, self.converter['method_impl'], self.converter, self.converter['method_str']
            )
        )

    def __exit__(self, type, val, tb):
        self._set_method(self.converter['method_impl'])

def default_input_names(num_inputs):
    return ["input_%d" % i for i in range(num_inputs)]

def default_output_names(num_outputs):
    return ["output_%d" % i for i in range(num_outputs)]


class LayerNamingNetworkWrapper(object):
    def __init__(self, ctx, network):
        self._ctx = ctx
        self._network = network
        self._layer_counts = defaultdict(lambda: 0)

    def _set_layer_name(self, layer):
        def arg_str(arg):
            if isinstance(arg, torch.Tensor):
                return "tensor(shape=%s, dtype=%s)" % (str(list(arg.shape)), str(arg.dtype))
            return str(arg)

        self._layer_counts[layer.type.name] += 1
        args = [arg_str(arg) for arg in self._ctx.method_args]
        kwargs = ["%s=%s" % (key, arg_str(arg)) for key, arg in self._ctx.method_kwargs.items()]
        layer.name = "[%s #%d] %s(%s)" % (layer.type.name, self._layer_counts[layer.type.name],
                                          self._ctx.method_str, ", ".join(args + kwargs))

    def __getattr__(self, name):
        attr = getattr(self._network, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                ret = attr(*args, **kwargs)
                if isinstance(ret, trt.ILayer):
                    self._set_layer_name(ret)
                return ret

            return wrapper
        else:
            return attr


class ConversionContext(object):
    def __init__(self, network, converters=CONVERTERS):
        self.network = LayerNamingNetworkWrapper(self, network)
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.hooks = [
            ConversionHook(self, key, converter)
            for key, converter in converters.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def add_inputs(self, torch_inputs, names=None):
        if names is None:
            names = default_input_names(len(torch_inputs))
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if not hasattr(torch_input, "_trt"):
                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=tuple(torch_input.shape),
                    dtype=torch_dtype_to_trt(torch_input.dtype),
                )
                trt_tensor.location = torch_device_to_trt(torch_input.device)
                torch_input._trt = trt_tensor

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = default_output_names(len(torch_outputs))
        self.output_names = names

        for i, torch_output in enumerate(torch_outputs):
            trt_tensor = torch_output._trt
            trt_tensor.name = names[i]
            trt_tensor.location = torch_device_to_trt(torch_output.device)
            trt_tensor.dtype = torch_dtype_to_trt(torch_output.dtype)
            self.network.mark_output(trt_tensor)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


def torch2trt(module, 
              inputs, 
              input_names=None, 
              output_names=None, 
              log_level=trt.Logger.ERROR, 
              max_batch_size=32,
              fp16_mode=False, 
              max_workspace_size=1<<30, 
              strict_type_constraints=False, 
              keep_network=True, 
              int8_mode=False, 
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              int8_calib_batch_size=1):

    inputs_in = inputs

    # copy inputs to avoid modifications to source data
    inputs = [tensor.clone() for tensor in inputs]  # only run single entry

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
        
    # run once to get num outputs
    outputs = module(*inputs)
    if not isinstance(outputs, tuple) and not isinstance(outputs, list):
        outputs = (outputs,)
        
    if input_names is None:
        input_names = default_input_names(len(inputs))
    if output_names is None:
        output_names = default_output_names(len(outputs))
        
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    with ConversionContext(network) as ctx:

        ctx.add_inputs(inputs, input_names)

        outputs = module(*inputs)

        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)

    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    builder.strict_type_constraints = strict_type_constraints

    if int8_mode:

        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)

        builder.int8_mode = True

        # @TODO(jwelsh):  Should we set batch_size=max_batch_size?  Need to investigate memory consumption
        builder.int8_calibrator = DatasetCalibrator(
            inputs, int8_calib_dataset, batch_size=int8_calib_batch_size, algorithm=int8_calib_algorithm
        )

    engine = builder.build_cuda_engine(network)

    module_trt = TRTModule(engine, input_names, output_names)

    if keep_network:
        module_trt.network = network

    return module_trt
