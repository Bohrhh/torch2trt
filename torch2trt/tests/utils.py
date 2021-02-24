import torch.nn as nn

MODULE_TESTS = []

# TEST MODULE
class TestInterface(nn.Module):
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
