import torch
import tensorrt as trt
from collections import defaultdict
from .utils import *

from .calibration import (
    TensorBatchDataset,
    DatasetCalibrator,
    DEFAULT_CALIBRATION_ALGORITHM,
)


# CONVERSION REGISTRY AND HOOKS

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

    def add_inputs(self, torch_inputs, names=None, dynamic_axes={}):
        if names is None:
            names = default_input_names(len(torch_inputs))
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if not hasattr(torch_input, "_trt"):
                trt_shape = list(torch_input.shape)
                for d in dynamic_axes:
                    trt_shape[d] = -1
                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=tuple(trt_shape),
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
            self.context.active_optimization_profile = 0
            
        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()

    def export(self, filename='model.trt'):
        with open(filename, 'wb') as f:
            f.write(self.engine.serialize())


def torch2trt(module, 
              inputs, 
              input_names=None, 
              output_names=None, 
              dynamic_axes={},
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
    """
    Args:
        dynamic_axes (dict): {dim1: [min1, max1],
                              dim2: [min2, max2]}
    """

    # ==================================================================
    # prepare input and output
    inputs_in = inputs

    # copy inputs to avoid modifications to source data
    inputs = [tensor.clone() for tensor in inputs]  # only run single entry

    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # run once to get num outputs
    with torch.no_grad():
        outputs = module(*inputs)
    if isinstance(outputs, list):
        outputs = tuple(outputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
        
    if input_names is None:
        input_names = default_input_names(len(inputs))
    if output_names is None:
        output_names = default_output_names(len(outputs))

    # ==================================================================
    # tensorrt objects
    logger  = trt.Logger(log_level)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config  = builder.create_builder_config()

    # ==================================================================
    # set tensorrt configs

    # dynamic shape profile
    if len(dynamic_axes)>0:
        profile = builder.create_optimization_profile()
        for i in range(len(inputs)):
            opt_shape = inputs[i].shape
            min_shape = list(opt_shape)
            max_shape = list(opt_shape)
            for d, v in dynamic_axes.items():
                min_shape[d] = v[0]
                max_shape[d] = v[1]
            profile.set_shape(input_names[i], min_shape, opt_shape, max_shape) 
        config.add_optimization_profile(profile)

    config.max_workspace_size = max_workspace_size
    if builder.platform_has_fast_fp16 and fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    builder.max_batch_size = max_batch_size
    builder.strict_type_constraints = strict_type_constraints

    # ==================================================================
    # construct tensorrt network
    with ConversionContext(network) as ctx:

        ctx.add_inputs(inputs, input_names, dynamic_axes=dynamic_axes)

        outputs = module(*inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)

    # ==================================================================
    # int8 calibration
    if int8_mode:

        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)

        builder.int8_mode = True

        # @TODO(jwelsh):  Should we set batch_size=max_batch_size?  Need to investigate memory consumption
        builder.int8_calibrator = DatasetCalibrator(
            inputs, int8_calib_dataset, batch_size=int8_calib_batch_size, algorithm=int8_calib_algorithm
        )

    # ==================================================================
    # construct tensorrt model
    engine = builder.build_engine(network, config)
    module_trt = TRTModule(engine, input_names, output_names)

    return module_trt
