from .split import convert_split
from torch2trt.utils import *


@tensorrt_converter('torch.chunk')
@tensorrt_converter('torch.Tensor.chunk')
def convert_chunk(ctx):
    convert_split(ctx)
    