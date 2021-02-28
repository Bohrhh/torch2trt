import torch
import torch.nn as nn
import torch.nn.functional as F
import torch2trt.plugins as plugins
from torch2trt.utils import trt_version
from .utils import *


# ========================================================================
# activation
# Relu activation: f(x) = x if x >= 0, f(x) = 0 if x < 0

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_relu_basic():
    return nn.ReLU()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
def test_functional_relu_basic():
    return TestInterface(lambda x: F.relu(x))


# ========================================================================
# activation
# Relu6 activation: f(x) = x if 0 <= x <= 6, f(x) = 0 if x < 0, f(x) = 6 if x > 6

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_relu6_basic():
    return torch.nn.ReLU6()
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
def test_functional_relu6_basic():
    return TestInterface(lambda x: F.relu6(x))


# ========================================================================
# activation
# Sigmoid activation

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_sigmoid_basic():
    return torch.nn.Sigmoid()


# ========================================================================
# activation
# Tanh

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_tanh_basic():
    return torch.nn.Tanh()


# ========================================================================
# activation
# Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_leaky_relu():
    return TestInterface(lambda x: F.leaky_relu(x))


# ========================================================================
# activation
# Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_elu():
    return TestInterface(lambda x: F.elu(x))


# ========================================================================
# activation
# Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_selu():
    return TestInterface(lambda x: F.selu(x))


# ========================================================================
# activation
# Softsign activation: f(x) = x / (1 + abs(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_softsign():
    return TestInterface(lambda x: F.softsign(x))


# ========================================================================
# activation
# Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_softplus():
    return TestInterface(lambda x: F.softplus(x))


# ========================================================================
# activation
# Hard sigmoid activation: f(x) = max(0, min(1, 1/6 * x + 0.5))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_hardsigmoid():
    return TestInterface(lambda x: F.hardsigmoid(x))


# ========================================================================
# adaptive_avg_pool2d
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_adaptive_avg_pool2d_1x1():
    return torch.nn.AdaptiveAvgPool2d((1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a')
def test_adaptive_avg_pool2d_2x2():
    return torch.nn.AdaptiveAvgPool2d((2, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 7  , 7  )], alphabet='a')
def test_adaptive_avg_pool2d_3x3():
    return torch.nn.AdaptiveAvgPool2d((3, 3))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 13 , 13 )], alphabet='a')
def test_adaptive_avg_pool2d_4x4():
    return torch.nn.AdaptiveAvgPool2d((4, 4))


# ========================================================================
# adaptive_max_pool2d
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_adaptive_max_pool2d_1x1():
    return torch.nn.AdaptiveMaxPool2d((1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='a')
def test_adaptive_max_pool2d_2x2():
    return torch.nn.AdaptiveMaxPool2d((2, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 7  , 7  )], alphabet='a')
def test_adaptive_max_pool2d_3x3():
    return torch.nn.AdaptiveMaxPool2d((3, 3))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 13 , 13 )], alphabet='a')
def test_adaptive_max_pool2d_4x4():
    return torch.nn.AdaptiveMaxPool2d((4, 4))


# ========================================================================
# add

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='a', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_basic():
    return TestInterface(lambda x, y: x+y)

class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='a', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_iadd():
    return IAdd()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='a')
def test_add_torch():
    return TestInterface(lambda x, y: torch.add(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='a')
def test_add_radd_int():
    return TestInterface(lambda x: 1 + x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='a', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_add_radd_float():
    return TestInterface(lambda x: 1.0 + x)

class AddConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x + self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='a')
def test_add_constant_nobatch():
    return AddConstantNoBatch()

class AddConstantBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x + self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='a')
def test_add_constant_batch():
    return AddConstantBatch()


# ========================================================================
# argmax

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='a')
def test_argmax_dim1():
    return TestInterface(lambda x: torch.argmax(x, dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmax_dim2():
    return TestInterface(lambda x: torch.argmax(x, dim=2, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
def test_argmax_dim3():
    return TestInterface(lambda x: torch.argmax(x, dim=3, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmax_keepdim():
    return TestInterface(lambda x: torch.argmax(x, dim=1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
def test_argmax_tensor():
    return TestInterface(lambda x: x.argmax(dim=1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmax_tensor_reduce():
    return TestInterface(lambda x: x.argmax())


# ========================================================================
# argmin

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='a')
def test_argmin_dim1():
    return TestInterface(lambda x: torch.argmin(x, dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],     alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],  alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmin_dim2():
    return TestInterface(lambda x: torch.argmin(x, dim=2, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
def test_argmin_dim3():
    return TestInterface(lambda x: torch.argmin(x, dim=3, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],    alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],     alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],  alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmin_keepdim():
    return TestInterface(lambda x: torch.argmin(x, dim=1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4)],     alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],  alphabet='a')
def test_argmin_tensor():
    return TestInterface(lambda x: x.argmin(dim=1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],  alphabet='a')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],  alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_argmin_tensor_reduce():
    return TestInterface(lambda x: x.argmin())


# ========================================================================
# avg_pool
# test avg_pool1d 

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a', dynamic_axes={0:[1,32], 2:[4,40]})
def test_avg_pool1d_k1s1p0():
    return torch.nn.AvgPool1d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool1d_k3s1p0():
    return torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool1d_k3s2p0():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool1d_k3s2p1():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool1d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)


# ========================================================================
# avg_pool
# test avg_pool2d

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[6,60]})
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a', dynamic_axes={0:[1,32], 2:[5,50], 3:[7,70]})
def test_avg_pool2d_k1s1p0():
    return torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool2d_k3s1p0():
    return torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool2d_k3s2p0():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool2d_k3s2p1():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool2d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)


# ========================================================================
# avg_pool
# test avg_pool3d 

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a', dynamic_axes={0:[1,32], 2:[4,40], 3:[6,60], 4:[8, 80]})
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a', dynamic_axes={0:[1,32], 2:[5,50], 3:[7,70], 4:[9, 90]})
def test_avg_pool3d_k1s1p0():
    return torch.nn.AvgPool3d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool3d_k3s1p0():
    return torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool3d_k3s2p0():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool3d_k3s2p1():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='a')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='a')
def test_avg_pool3d_k3s2p1_with_ceil_mode():
    return torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)


# ========================================================================
# batch_norm

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)],       enabled=trt_version() >= '7.0', alphabet='b')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)],       enabled=trt_version() >= '7.0', alphabet='b', dynamic_axes={0:[1,32], 2:[3,30]})
def test_batch_norm_1d():
    return torch.nn.BatchNorm1d(10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4)],    enabled=trt_version() >= '7.0', alphabet='b')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4)],    enabled=trt_version() >= '7.0', alphabet='b', dynamic_axes={0:[1,32], 2:[3,30], 3:[4,40]})
def test_batch_norm_2d():
    return torch.nn.BatchNorm2d(10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4, 5)], enabled=trt_version() >= '7.0', alphabet='b')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 4, 5)], enabled=trt_version() >= '7.0', alphabet='b', dynamic_axes={0:[1,32], 2:[3,30], 3:[4,40], 4:[5,50]})
def test_batch_norm_3d():
    return torch.nn.BatchNorm3d(10)


# ========================================================================
# cat

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)], alphabet='c', dynamic_axes={0:[1,32], 1:[1, 40]})
def test_cat_basic():
    return TestInterface(lambda *x: torch.cat(x, dim=1))


# ========================================================================
# chunk

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='c')
def test_torch_chunk_c1d1():
    return TestInterface(lambda x: torch.chunk(x, 1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='c')
def test_torch_chunk_c2d1():
    return TestInterface(lambda x: torch.chunk(x, 2, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='c')
def test_torch_chunk_c3d1():
    return TestInterface(lambda x: torch.chunk(x, 3, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='c')
def test_torch_chunk_c3d2():
    return TestInterface(lambda x: torch.chunk(x, 3, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='c')
def test_tensor_chunk_c3d2():
    return TestInterface(lambda x: x.chunk(3, 2))


# ========================================================================
# clamp
# clamp_min

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp_min():
    return TestInterface(lambda x: torch.clamp_min(x, -0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_min():
    return TestInterface(lambda x: x.clamp_min(-0.1))


# ========================================================================
# clamp
# clamp_max

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp_max():
    return TestInterface(lambda x: torch.clamp_max(x, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_max():
    return TestInterface(lambda x: x.clamp_max(0.1))


# ========================================================================
# clamp

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp():
    return TestInterface(lambda x: torch.clamp(x, -0.1, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_tensor_clamp():
    return TestInterface(lambda x: x.clamp(-0.1, 0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp_option_max():
    return TestInterface(lambda x: torch.clamp(x, max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp_option_min():
    return TestInterface(lambda x: torch.clamp(x, min=-0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_torch_clamp_option_max_min():
    return TestInterface(lambda x: torch.clamp(x, min=-0.1, max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_tensor_clamp_option_max():
    return TestInterface(lambda x: x.clamp(max=0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
def test_tensor_clamp_option_min():
    return TestInterface(lambda x: x.clamp(min=-0.1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='c', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_tensor_clamp_max_min():
    return TestInterface(lambda x: x.clamp(min=-0.1, max=0.1))


# ========================================================================
# compare

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0', alphabet='c', dynamic_axes={0:[1,32], 2:[6,60], 3:[6,60]})
def test_gt_tensor():
    return TestInterface(lambda x, y: x>y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)],               enabled=trt_version() >= '7.0', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)],               enabled=trt_version() >= '7.0', alphabet='c', dynamic_axes={0:[1,32], 2:[6,60], 3:[6,60]})
def test_gt_float():
    return TestInterface(lambda x: x>0.1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0', alphabet='c')
def test_lt_tensor():
    return TestInterface(lambda x, y: x<y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)],               enabled=trt_version() >= '7.0', alphabet='c')
def test_lt_float():
    return TestInterface(lambda x: x<0.1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0', alphabet='c')
def test_eq_tensor():
    return TestInterface(lambda x, y: x==y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)],               enabled=trt_version() >= '7.0', alphabet='c')
def test_eq_float():
    return TestInterface(lambda x: x==0.1)


# ========================================================================
# conv
# test conv1d 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c', dynamic_axes={0:[1,32], 2:[128,256]})
def test_conv1d_k1s1p0d1():
    return torch.nn.Conv1d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv1d_k3s1p0d1():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv1d_k3s2p0d1():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv1d_k3s2p1d1():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv1d_k3s2p1d2():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv1d_k3s2p1d2_nobias():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# conv
# test conv2d 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_conv2d_k1s1p0d1():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv2d_k3s1p0d1():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv2d_k3s2p0d1():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv2d_k3s2p1d1():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv2d_k3s2p1d2():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv2d_k3s2p1d2_nobias():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# conv
# test conv3d 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c', dynamic_axes={0:[1,32], 2:[64,100], 3:[64,100], 4:[64,100]})
def test_conv3d_k1s1p0d1():
    return torch.nn.Conv3d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv3d_k3s1p0d1():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv3d_k3s2p0d1():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv3d_k3s2p1d1():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv3d_k3s2p1d2():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0', alphabet='c')
def test_conv3d_k3s2p1d2_nobias():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# conv_transpose
# test ConvTranspose1d 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c', dynamic_axes={0:[1,32], 2:[128,256]})
def test_ConvTranspose1d_k1s1p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose1d_k3s1p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose1d_k3s2p0d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose1d_k3s2p1d1():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose1d_k3s2p1d2():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose1d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose1d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# conv_transpose
# test ConvTranspose2d 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_ConvTranspose2d_k1s1p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose2d_k3s1p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose2d_k3s2p0d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose2d_k3s2p1d1():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose2d_k3s2p1d2():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose2d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# conv_transpose
# test ConvTranspose3d
 
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c', dynamic_axes={0:[1,32], 2:[32,64], 3:[32,64], 4:[32,64]})
def test_ConvTranspose3d_k1s1p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=1, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose3d_k3s1p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=1, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose3d_k3s2p0d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=0, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose3d_k3s2p1d1():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose3d_k3s2p1d2():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.1.3', alphabet='c')
def test_ConvTranspose3d_k3s2p1d2_nobias():
    return torch.nn.ConvTranspose3d(10, 5, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)


# ========================================================================
# correlation

@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c', dynamic_axes={0:[1,32], 2:[64,128], 3:[64,128]})
def test_correlation_time_mean():
    return plugins.Correlation_TRT(max_disp=3, stride=1, mode='time', reduction='mean')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c')
def test_correlation_l1_mean():
    return plugins.Correlation_TRT(max_disp=3, stride=1, mode='l1', reduction='mean')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c', dynamic_axes={0:[1,32], 2:[64,128], 3:[64,128]})
def test_correlation_l1_mean_s2():
    return plugins.Correlation_TRT(max_disp=3, stride=2, mode='l1', reduction='mean')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c')
def test_correlation_time_sum():
    return plugins.Correlation_TRT(max_disp=5, stride=1, mode='time', reduction='sum')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 64, 64), (1, 32, 64, 64)], alphabet='c')
def test_correlation_l1_sum():
    return plugins.Correlation_TRT(max_disp=5, stride=1, mode='l1', reduction='sum')

# ========================================================================
# div

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='d')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5), (1, 3, 4, 5)],         alphabet='d', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_div_basic():
    return TestInterface(lambda x, y: x / y)

class IDiv(torch.nn.Module):
    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='d')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5), (1, 3, 4, 5)],         alphabet='d', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_div_idiv():
    return IDiv()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='d')
def test_div_torch():
    return TestInterface(lambda x, y: torch.div(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='d')
def test_rdiv_int():
    return TestInterface(lambda x: 1 / x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='d')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],                       alphabet='d', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_rdiv_float():
    return TestInterface(lambda x: 1.0 / x)

class DivConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(DivConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x / self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='d')
def test_div_constant_nobatch():
    return DivConstantNoBatch()

class DivConstantBatch(torch.nn.Module):
    def __init__(self):
        super(DivConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x / self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='d')
def test_div_constant_batch():
    return DivConstantBatch()


# ========================================================================
# expand

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)], alphabet='e')
def test_tensor_expand_singledim():
    return TestInterface(lambda x: x.expand(1,3,3,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)], alphabet='e')
def test_tensor_expand_multidim():
    return TestInterface(lambda x: x.expand(1,3,3,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)], alphabet='e')
def test_tensor_expand_inferdim():
    return TestInterface(lambda x: x.expand(1,3,-1,-1))


# ========================================================================
# getitem

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='g')
def test_tensor_getitem_1d_int():
    return TestInterface(lambda x: x[:, 0])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='g')
def test_tensor_getitem_1slice():
    return TestInterface(lambda x: x[0])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_int():
    return TestInterface(lambda x: x[:, 0])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_strided():
    return TestInterface(lambda x: x[:, ::2])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_strided_offset():
    return TestInterface(lambda x: x[:, 1::2])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_strided_range():
    return TestInterface(lambda x: x[:, 1:3:2])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_insert_dim():
    return TestInterface(lambda x: x[:, None])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return TestInterface(lambda x: x[:, None, ...])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_append_dim():
    return TestInterface(lambda x: x[:, ..., None])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_append_2dim():
    return TestInterface(lambda x: x[:, ..., None, None])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)], alphabet='g')
def test_tensor_getitem_2d_weird_combo():
    return TestInterface(lambda x: x[:, 0:3:4, None, None, 1, ...])


# ========================================================================
# grid_sample

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112), (1, 32, 32, 2)],        alphabet='g')
def test_grid_sample_2d():
    return TestInterface(lambda x,y: F.grid_sample(x,y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112), (1, 32, 32, 2)],        alphabet='g')
def test_grid_sample_2d_align_corners():
    return TestInterface(lambda x,y: F.grid_sample(x, y, align_corners=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112), (1, 32, 32, 2)],        alphabet='g')
def test_grid_sample_2d_nearest():
    return TestInterface(lambda x,y: F.grid_sample(x, y, mode='nearest'))


# ========================================================================
# group_norm

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112)],      enabled=trt_version() >= '7.1.3', alphabet='g')
def test_group_norm_g2_1d():
    return torch.nn.GroupNorm(2, 10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], enabled=trt_version() >= '7.1.3', alphabet='g')
def test_group_norm_g2_2d():
    return torch.nn.GroupNorm(2, 10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112)],      enabled=trt_version() >= '7.1.3', alphabet='g')
def test_group_norm_g2_eps_1d():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], enabled=trt_version() >= '7.1.3', alphabet='g')
def test_group_norm_g2_eps_2d():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)


# ========================================================================
# identity

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)],   alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)],   alphabet='i', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_tensor_contiguous():
    return TestInterface(lambda x: (x+1).contiguous())

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3)],     alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)],   alphabet='i')
def test_dropout():
    return TestInterface(lambda x: F.dropout((x+1),   training=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)],   alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)],   alphabet='i', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_dropout2d():
    return TestInterface(lambda x: F.dropout2d((x+1), training=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3,3)], alphabet='i')
def test_dropout3d():
    return TestInterface(lambda x: F.dropout3d((x+1), training=False))


# ========================================================================
# instance_norm

# STATIC

# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# def test_instance_norm_1d_static():
#     return torch.nn.InstanceNorm1d(10, track_running_stats=True)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
# def test_instance_norm_2d_static():
#     return torch.nn.InstanceNorm2d(10, track_running_stats=True)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
# def test_instance_norm_3d_static():
#     return torch.nn.InstanceNorm3d(10, track_running_stats=True)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# def test_instance_norm_1d_static_affine():
#     return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=True)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
# def test_instance_norm_2d_static_affine():
#     return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=True)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
# def test_instance_norm_3d_static_affine():
#     return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=True)

# # DYNAMIC

# # @TODO(jwelsh): 1D dynamic test failing
# # @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# # def test_instance_norm_1d_dynamic():
# #     return torch.nn.InstanceNorm1d(10, track_running_stats=False)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
# def test_instance_norm_2d_dynamic():
#     return torch.nn.InstanceNorm2d(10, track_running_stats=False)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
# def test_instance_norm_3d_dynamic():
#     return torch.nn.InstanceNorm3d(10, track_running_stats=False)


# # @TODO(jwelsh): 1D dynamic test failing
# # @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# # def test_instance_norm_1d_dynamic_affine():
# #     return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=False)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
# def test_instance_norm_2d_dynamic_affine():
#     return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=False)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
# def test_instance_norm_3d_dynamic_affine():
#     return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=False)


# ========================================================================
# interpolate

@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_nearest():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,5,13,13)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_bilinear():
    return torch.nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,12,12)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_size():
    return torch.nn.Upsample(size=3, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,13,13)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1)],      enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)],    enabled=trt_version() >= '7.1', alphabet='i', dynamic_axes={0:[1,32], 2:[12,48], 3:[12,48]})
def test_interpolate_size_odd_input():
    return torch.nn.Upsample(size=[24,36], mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,6,6,6)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_nearest_3d():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,6,7,7,7)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,2,4,4)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1,1)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,5,5,5)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_bilinear_3d():
    return torch.nn.Upsample(scale_factor=3, mode="trilinear", align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,8,8,8)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_align_corner_3d():
    return torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)], enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_size_3d():
    return torch.nn.Upsample(size=3, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,7,9,5)],    enabled=trt_version() >= '7.1', alphabet='i')
@add_module_test(torch.float32, torch.device('cuda'), [(1,4,3,5,1)],    enabled=trt_version() >= '7.1', alphabet='i')
def test_interpolate_size_odd_input_3d():
    return torch.nn.Upsample(size=[11,14,17], mode="trilinear", align_corners=False)


# ========================================================================
# linear

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l', dynamic_axes={0:[1,32]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l', dynamic_axes={0:[1,32], 1:[3,30], 2:[4,40]})
def test_linear_basic():
    return torch.nn.Linear(10, 5)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
def test_linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)


# ========================================================================
# log_softmax

# dim==0 has some unknow error
# @add_module_test(torch.float32, torch.device('cuda'), [(3, 4, 10)])
# def test_logsoftmax_d0():
#     return nn.LogSoftmax(dim=0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l', dynamic_axes={0:[1,32], 1:[10,50]})
def test_logsoftmax_d1():
    return nn.LogSoftmax(dim=1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
def test_tensor_logsoftmax_d1():
    return TestInterface(lambda x: x.log_softmax(dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)],       alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
def test_torch_logsoftmax_d1():
    return TestInterface(lambda x: torch.log_softmax(x, dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)], alphabet='l')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)],    alphabet='l', dynamic_axes={0:[1,32], 1:[3,30], 2:[10,50]})
def test_logsoftmax_d2():
    return nn.LogSoftmax(dim=2)


# ========================================================================
# max

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_max_dim1():
    return TestInterface(lambda x: torch.max(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
def test_max_dim2():
    return TestInterface(lambda x: torch.max(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_max_dim1_keepdim():
    return TestInterface(lambda x: torch.max(x, 1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_max_reduce_all():
    return TestInterface(lambda x: torch.max(x))
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)],         alphabet='m') # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)], alphabet='m') # broadcast
def test_max_elementwise():
    return TestInterface(lambda x, y: torch.max(x,y))


# ========================================================================
# max_pool
# test max_pool1d

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m', dynamic_axes={0:[1,32], 2:[4,40]})
def test_max_pool1d_k1s1p0():
    return torch.nn.MaxPool1d(kernel_size=1, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool1d_k3s1p0():
    return torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool1d_k3s2p0():
    return torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool1d_k3s2p1():
    return torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool1d_k3s2p1_with_ceil_mode():
    return torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)


# ========================================================================
# max_pool
# test max_pool2d

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m', dynamic_axes={0:[1,32], 2:[4,40], 3:[6,60]})
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m', dynamic_axes={0:[1,32], 2:[5,50], 3:[7,70]})
def test_max_pool2d_k1s1p0():
    return torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool2d_k3s1p0():
    return torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool2d_k3s2p0():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool2d_k3s2p1():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool2d_k3s2p1_with_ceil_mode():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)


# ========================================================================
# max_pool
# test max_pool3d

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m', dynamic_axes={0:[1,32], 2:[4,40], 3:[6,60], 4:[8, 80]})
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m', dynamic_axes={0:[1,32], 2:[5,50], 3:[7,70], 4:[9, 90]})
def test_max_pool3d_k1s1p0():
    return torch.nn.MaxPool3d(kernel_size=1, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool3d_k3s1p0():
    return torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool3d_k3s2p0():
    return torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool3d_k3s2p1():
    return torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6, 8)], enabled=trt_version() >= '7.0', alphabet='m')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 5, 7, 9)], enabled=trt_version() >= '7.0', alphabet='m')
def test_max_pool3d_k3s2p1_with_ceil_mode():
    return torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True)


# ========================================================================
# mean

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_torch_d1():
    return TestInterface(lambda x: torch.mean(x, dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_tensor_d1():
    return TestInterface(lambda x: x.mean(dim=1, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_mean_torch_d2():
    return TestInterface(lambda x: torch.mean(x, dim=2, keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_tensor_reduce():
    return TestInterface(lambda x: x.mean())

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_torch_d1_d2():
    return TestInterface(lambda x: torch.mean(x, dim=(1,2), keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_tensor_d1_d2():
    return TestInterface(lambda x: x.mean(dim=[1, 2], keepdim=False))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='m')
def test_mean_torch_keepdim():
    return TestInterface(lambda x: torch.mean(x, dim=1, keepdim=True))




# ========================================================================
# min

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_min_reduce_dim1():
    return TestInterface(lambda x: torch.min(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
def test_min_reduce_dim2():
    return TestInterface(lambda x: torch.min(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_min_reduce_dim1_keepdim():
    return TestInterface(lambda x: torch.min(x, 1, keepdim=True))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],                  alphabet='m', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],               alphabet='m', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_min_reduce_all():
    return TestInterface(lambda x: torch.min(x))
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)],    alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)],         alphabet='m') # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)], alphabet='m') # broadcast
def test_min_elementwise():
    return TestInterface(lambda x, y: torch.min(x,y))


# ========================================================================
# mul

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='m', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_mul_basic():
    return TestInterface(lambda x, y: x*y)

class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='m', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_mul_imul():
    return IMul()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='m')
def test_mul_torchmul():
    return TestInterface(lambda x, y: torch.mul(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 244, 244)],                   alphabet='m')
def test_rmul_int():
    return TestInterface(lambda x: 10*x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 244, 244)],                   alphabet='m')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 244, 244)],                   alphabet='m', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_rmul_float():
    return TestInterface(lambda x: 10.0*x)

class MulConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x * self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='m')
def test_mul_constant_nobatch():
    return MulConstantNoBatch()

class MulConstantBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x * self.y
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='m')
def test_mul_constant_batch():
    return MulConstantBatch()


# ========================================================================
# narrow

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)], alphabet='n')
def test_narrow1():
    return TestInterface(lambda x: torch.narrow(x, 1, 0, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)], alphabet='n')
def test_narrow2():
    return TestInterface(lambda x: torch.narrow(x, 2, 2, 50))


# ========================================================================
# normalize

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='n')
def test_normalize_basic():
    return TestInterface(lambda x: F.normalize(x))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='n')
def test_normalize_l1_basic():
    return TestInterface(lambda x: F.normalize(x, p=1.0))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='n', dynamic_axes={0:[1,32], 1:[3,30]})
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='n', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30]})
def test_normalize_l1p5_basic():
    return TestInterface(lambda x: F.normalize(x, p=1.5))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='n')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='n')
def test_normalize_l2_height():
    return TestInterface(lambda x: F.normalize(x, p=2.0, dim=2))


# ========================================================================
# pad

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='p', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_pad_basic():
    return TestInterface(lambda x: F.pad(x, (1, 2, 3, 4)))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], alphabet='p', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_pad_last():
    return TestInterface(lambda x: F.pad(x, (1, 2)))


# ========================================================================
# permute

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],    alphabet='p')
def test_permute_2d_0123():
    return TestInterface(lambda x: x.permute(0,1,2,3))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],    alphabet='p')
def test_permute_2d_0312():
    return TestInterface(lambda x: x.permute(0,3,1,2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],    alphabet='p')
def test_permute_2d_3012():
    return TestInterface(lambda x: x.permute(3,0,2,1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)], alphabet='p')
def test_permute_3d_01234():
    return TestInterface(lambda x: x.permute(0,1,2,3,4))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)], alphabet='p')
def test_permute_3d_04132():
    return TestInterface(lambda x: x.permute(0,4,1,3,2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)], alphabet='p')
def test_permute_list():
    return TestInterface(lambda x: x.permute([0,4,1,3,2]))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)], alphabet='p')
def test_permute_tuple():
    return TestInterface(lambda x: x.permute((0,4,1,3,2)))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)],    alphabet='p', dynamic_axes={0:[1,32], 2:[4,40], 3:[4,40]})
def test_permute_2d_0132_dynamic():
    return TestInterface(lambda x: x.permute(0,1,3,2))


# ========================================================================
# pow

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='p', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_pow_basic():
    return TestInterface(lambda x, y: x**y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='p')
def test_torch_pow():
    return TestInterface(lambda x, y: torch.pow(x,y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='p')
def test_rpow_int():
    return TestInterface(lambda x: 2**x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='p', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_rpow_float():
    return TestInterface(lambda x: 2.0**x)


# ========================================================================
# prelu

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5)],       alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)], alphabet='p', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_prelu_scalar():
    return torch.nn.PReLU()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5)],       alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)], alphabet='p', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_prelu_vector():
    m = torch.nn.PReLU(5)
    m.weight = torch.nn.Parameter(torch.randn(5)) # randn so each channel different
    return m


# ========================================================================
# prod

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p', dynamic_axes={0:[1,32], 2:[3,30]})
def test_prod_reduce_all():
    return TestInterface(lambda x: torch.prod(x))     

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p')
def test_prod_reduce_dim1():
    return TestInterface(lambda x: torch.prod(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p', dynamic_axes={0:[1,32], 2:[3,30]})
def test_prod_reduce_dim2():
    return TestInterface(lambda x: torch.prod(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='p')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='p')
def test_prod_reduce_dim1_keepdim():
    return TestInterface(lambda x: torch.prod(x, 1, keepdim=True))


# ========================================================================
# size

# dynamic_axes should be provided in this test example
# or the input binding can not be found
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='s', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_size_basic_dynamic():
    return TestInterface(lambda x: x.size())

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='s', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_size_dim2_dynamic():
    return TestInterface(lambda x: x.size(2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='s', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_shape_basic_dynamic():
    return TestInterface(lambda x: x.shape)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)], alphabet='s', dynamic_axes={0:[1,32], 2:[4,40], 3:[5,50]})
def test_shape_dim0_dynamic():
    return TestInterface(lambda x: x.shape[0])


# ========================================================================
# softmax

@add_module_test(torch.float32, torch.device('cuda'), [(4, 3, 3, 3)], alphabet='s')
def test_softmax_dim0():
    return torch.nn.Softmax(0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(4, 3, 3, 3)], alphabet='s', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_softmax_dim1():
    return torch.nn.Softmax(1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_softmax_dim2():
    return torch.nn.Softmax(2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],       alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_softmax_dim_neg1():
    return torch.nn.Softmax(-1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_softmax_dim_neg2():
    return torch.nn.Softmax(-2)


# ========================================================================
# split

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)], alphabet='s')
def test_torch_split_1_d0():
    return TestInterface(lambda x: torch.split(x, 1, 0))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_torch_split_1_d1():
    return TestInterface(lambda x: torch.split(x, 1, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_torch_split_2_d1():
    return TestInterface(lambda x: torch.split(x, 2, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_torch_split_3_d1():
    return TestInterface(lambda x: torch.split(x, 3, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_torch_split_1_d2():
    return TestInterface(lambda x: x.split(1, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], alphabet='s')
def test_tensor_split_2_d2():
    return TestInterface(lambda x: x.split(2, 2))


# ========================================================================
# stack

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0', alphabet='s')
def test_stack_dim1():
    return TestInterface(lambda *x: torch.stack(x, dim=1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0', alphabet='s')
def test_stack_dim3():
    return TestInterface(lambda *x: torch.stack(x, dim=3))


# ========================================================================
# sub

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='s', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_sub_basic():
    return TestInterface(lambda x, y: x-y)

class ISub(torch.nn.Module):
    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='s', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_sub_isub():
    return ISub()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)], alphabet='s')
def test_torch_sub():
    return TestInterface(lambda x, y: torch.sub(x, y))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='s')
def test_rsub_int():
    return TestInterface(lambda x: 1-x)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)],                   alphabet='s', dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def test_rsub_float():
    return TestInterface(lambda x: 1.0-x)

class SubConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x - self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='s')
def test_sub_constant_nobatch():
    return SubConstantNoBatch()

class SubConstantBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x - self.y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)],                     alphabet='s')
def test_sub_constant_batch():
    return SubConstantBatch()


# ========================================================================
# sum

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s', dynamic_axes={0:[1,32], 2:[3,30]})
def test_sum_reduce_all():
    return TestInterface(lambda x: torch.sum(x))     

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s', dynamic_axes={0:[1,32], 2:[3,30]})
def test_sum_dim1():
    return TestInterface(lambda x: torch.sum(x, 1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s')
def test_sum_dim2():
    return TestInterface(lambda x: torch.sum(x, 2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],    alphabet='s')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)], alphabet='s')
def test_sum_dim1_keepdim():
    return TestInterface(lambda x: torch.sum(x, 1, keepdim=True))


# ========================================================================
# transpose

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)],    alphabet='t')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)], alphabet='t')
def test_torch_transpose_02():
    return TestInterface(lambda x: torch.transpose(x, 0, 2))

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)],    alphabet='t')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)], alphabet='t')
def test_tensor_transpose_02():
    return TestInterface(lambda x: x.transpose(0, 2))

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)],    alphabet='t')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)], alphabet='t')
def test_tensor_transpose_12():
    return TestInterface(lambda x: x.transpose(1, 2))

@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)], alphabet='t')
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)], alphabet='t', dynamic_axes={0:[1,32], 2:[3,30], 3:[3,30]})
def test_tensor_transpose_23():
    return TestInterface(lambda x: x.transpose(2, 3))


# ========================================================================
# unary
# exp

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_exp():
    return TestInterface(lambda x: torch.exp(x))


# ========================================================================
# unary
# log

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_log():
    return TestInterface(lambda x: torch.log(x))


# ========================================================================
# unary
# sqrt

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_sqrt():
    return TestInterface(lambda x: torch.sqrt(x))


# ========================================================================
# unary
# reciprocal

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_reciprocal():
    return TestInterface(lambda x: torch.reciprocal(x))


# ========================================================================
# unary
# abs

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_abs():
    return TestInterface(lambda x: torch.abs(x))


# ========================================================================
# unary
# neg

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_neg():
    return TestInterface(lambda x: torch.neg(x))


# ========================================================================
# unary
# sin

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_sin():
    return TestInterface(lambda x: torch.sin(x))


# ========================================================================
# unary
# cos

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_cos():
    return TestInterface(lambda x: torch.cos(x))


# ========================================================================
# unary
# tan

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_tan():
    return TestInterface(lambda x: torch.tan(x))


# ========================================================================
# unary
# sinh

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_sinh():
    return TestInterface(lambda x: torch.sinh(x))


# ========================================================================
# unary
# cosh

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_cosh():
    return TestInterface(lambda x: torch.cosh(x))


# ========================================================================
# unary
# asin

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_asin():
    return TestInterface(lambda x: torch.asin(x))


# ========================================================================
# unary
# acos

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_acos():
    return TestInterface(lambda x: torch.acos(x))


# ========================================================================
# unary
# atan

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_atan():
    return TestInterface(lambda x: torch.atan(x))


# ========================================================================
# unary
# ceil

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_ceil():
    return TestInterface(lambda x: torch.ceil(x))


# ========================================================================
# unary
# floor

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)], alphabet='u')
def test_floor():
    return TestInterface(lambda x: torch.floor(x))


# ========================================================================
# view

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
def test_flatten_1d_fake():
    return TestInterface(lambda x: x.flatten(1, 1)+1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_flatten_1d():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)],       alphabet='v')
def test_flatten_batch2_1d():
    return TestInterface(lambda x: x.flatten(1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_view_1d():
    return TestInterface(lambda x: x.view(1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 3, 3)],       alphabet='v')
def test_view_batch2_1d():
    return TestInterface(lambda x: x.view(2, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
def test_view_2d():
    return TestInterface(lambda x: x.view(1, 1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 6)],    alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)], alphabet='v')
def test_view_3d():
    return TestInterface(lambda x: x.view(1, 3, 3, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_view_size():
    return TestInterface(lambda x: x.view(x.size(0), -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_tensor_reshape_1d():
    return TestInterface(lambda x: x.reshape(1, -1))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_torch_reshape_1d():
    return TestInterface(lambda x: torch.reshape(x, (1,-1)))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 7)],             alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)],       alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 2:[3,30], 3:[3,30]})
def test_unsqueeze():
    return TestInterface(lambda x: x.unsqueeze(dim=2))

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1)],          alphabet='v')
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)],       alphabet='v', dynamic_axes={0:[1,32], 1:[3,30], 3:[3,30]})
def test_squeeze():
    return TestInterface(lambda x: x.squeeze(dim=2))
