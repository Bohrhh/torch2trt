# dummy converters throw warnings method encountered
import tensorrt as trt
from .dummy_converters import *

# supported converters will override dummy converters

from .activation import *
from .adaptive_avg_pool2d import *
from .adaptive_max_pool2d import *
from .add import *
from .argmax import *
from .argmin import *
from .avg_pool import *
from .batch_norm import *
from .cat import *
from .chunk import *
from .clamp import *
from .compare import *
from .conv import *
from .conv_transpose import *
from .correlation import *
from .div import *
from .expand import *
from .gather import *
from .getitem import *
from .grid_sample import *
from .group_norm import *
from .identity import *
# from .instance_norm import *
from .interpolate import *
from .linear import *
from .log_softmax import *
from .matmul import *
from .max import *
from .max_pool import *
from .mean import *
from .min import *
from .mul import *
from .narrow import *
from .normalize import *
from .pad import *
from .permute import *
from .pow import *
from .prelu import *
from .prod import *
from .size import *
from .softmax import *
from .split import *
from .stack import *
from .sub import *
from .sum import *
from .topk import *
from .transpose import *
from .unary import *
from .view import *
