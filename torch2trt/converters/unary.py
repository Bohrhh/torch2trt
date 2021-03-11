from torch2trt.utils import *

        
def __convert_unary(ctx, op):
    # parse args
    input  = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return

    # get tensorrt input
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # add tensorrt layer
    layer = ctx.network.add_unary(input_trt, op)

    # get tensorrt output
    output._trt = layer.get_output(0)


# ABS : Absolute value
@tensorrt_converter('torch.abs')
@tensorrt_converter('torch.abs_')
@tensorrt_converter('torch.Tensor.abs')
@tensorrt_converter('torch.Tensor.abs_')
def convert_abs(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ABS)


# ACOS : Inverse cosine
@tensorrt_converter('torch.acos')
@tensorrt_converter('torch.acos_')
@tensorrt_converter('torch.Tensor.acos')
@tensorrt_converter('torch.Tensor.acos_')
def convert_acos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ACOS)


# ASIN : Inverse sine
@tensorrt_converter('torch.asin')
@tensorrt_converter('torch.asin_')
@tensorrt_converter('torch.Tensor.asin')
@tensorrt_converter('torch.Tensor.asin_')
def convert_asin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ASIN)


# ATAN : Inverse tangent
@tensorrt_converter('torch.atan')
@tensorrt_converter('torch.atan_')
@tensorrt_converter('torch.Tensor.atan')
@tensorrt_converter('torch.Tensor.atan_')
def convert_atan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ATAN)


# CEIL : Ceiling
@tensorrt_converter('torch.ceil')
@tensorrt_converter('torch.ceil_')
@tensorrt_converter('torch.Tensor.ceil')
@tensorrt_converter('torch.Tensor.ceil_')
def convert_ceil(ctx):
    __convert_unary(ctx, trt.UnaryOperation.CEIL)


#  COS : Cosine
@tensorrt_converter('torch.cos')
@tensorrt_converter('torch.cos_')
@tensorrt_converter('torch.Tensor.cos')
@tensorrt_converter('torch.Tensor.cos_')
def convert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COS)


# COSH : Hyperbolic cosine
@tensorrt_converter('torch.cosh')
@tensorrt_converter('torch.cosh_')
@tensorrt_converter('torch.Tensor.cosh')
@tensorrt_converter('torch.Tensor.cosh_')
def convert_cosh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COSH)


# EXP : Exponentiation
@tensorrt_converter('torch.exp')
@tensorrt_converter('torch.exp_')
@tensorrt_converter('torch.Tensor.exp')
@tensorrt_converter('torch.Tensor.exp_')
def convert_exp(ctx):
    __convert_unary(ctx, trt.UnaryOperation.EXP)


# FLOOR : Floor
@tensorrt_converter('torch.floor')
@tensorrt_converter('torch.floor_')
@tensorrt_converter('torch.Tensor.floor')
@tensorrt_converter('torch.Tensor.floor_')
def convert_floor(ctx):
    __convert_unary(ctx, trt.UnaryOperation.FLOOR)


# LOG : Log (base e)
@tensorrt_converter('torch.log')
@tensorrt_converter('torch.log_')
@tensorrt_converter('torch.Tensor.log')
@tensorrt_converter('torch.Tensor.log_')
def convert_log(ctx):
    __convert_unary(ctx, trt.UnaryOperation.LOG)


# NEG : Negation
@tensorrt_converter('torch.neg')
@tensorrt_converter('torch.neg_')
@tensorrt_converter('torch.Tensor.neg')
@tensorrt_converter('torch.Tensor.__neg__')
@tensorrt_converter('torch.Tensor.neg_')
def convert_neg(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NEG)


# NOT : not
@tensorrt_converter('torch.Tensor.__invert__')
def convert_not(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NOT)


# RECIP : Reciprocal
@tensorrt_converter('torch.reciprocal')
@tensorrt_converter('torch.reciprocal_')
@tensorrt_converter('torch.Tensor.reciprocal')
@tensorrt_converter('torch.Tensor.reciprocal_')
def convert_reciprocal(ctx):
    __convert_unary(ctx, trt.UnaryOperation.RECIP)


# SIN : Sine
@tensorrt_converter('torch.sin')
@tensorrt_converter('torch.sin_')
@tensorrt_converter('torch.Tensor.sin')
@tensorrt_converter('torch.Tensor.sin_')
def convert_sin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SIN)


# SINH : Hyperbolic sine
@tensorrt_converter('torch.sinh')
@tensorrt_converter('torch.sinh_')
@tensorrt_converter('torch.Tensor.sinh')
@tensorrt_converter('torch.Tensor.sinh_')
def convert_sinh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SINH)


# SQRT : Square root
@tensorrt_converter('torch.sqrt')
@tensorrt_converter('torch.sqrt_')
@tensorrt_converter('torch.Tensor.sqrt')
@tensorrt_converter('torch.Tensor.sqrt_')
def convert_sqrt(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SQRT)


# TAN : Tangent
@tensorrt_converter('torch.tan')
@tensorrt_converter('torch.tan_')
@tensorrt_converter('torch.Tensor.tan')
@tensorrt_converter('torch.Tensor.tan_')
def convert_tan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.TAN)





#  |    ASINH : Inverse hyperbolic sine
#  |  
#  |    ACOSH : Inverse hyperbolic cosine
#  |  
#  |    ATANH : Inverse hyperbolic tangent


