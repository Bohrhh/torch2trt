import torch
import torch.nn as nn
import torch.nn.functional as F
from torch2trt.opts.dcn import ModulatedDeformConvFunction
from torch2trt.opts.dcn import ModulatedDeformConvPack

class Correlation_TRT(nn.Module):
    def __init__(self, max_disp, stride=1, mode='time', reduction='mean'):
        super(Correlation_TRT, self).__init__()
        self.max_disp = max_disp
        self.stride = stride
        self.mode = mode
        self.reduction = reduction

    def forward(self, left, right):
        cost_tensors = []
        w = right.shape[-1]
        right = F.pad(right, (self.max_disp, self.max_disp))
        for i in range(-self.max_disp, self.max_disp+1, self.stride):
            if self.mode == 'time':
                t = left*right[:,:,:,i+self.max_disp:i+w+self.max_disp]
            else:
                t = (left-right[:,:,:,i+self.max_disp:i+w+self.max_disp]).abs()

            t = t.mean(dim=1, keepdims=True) if self.reduction=='mean' else t.sum(dim=1, keepdims=True)
            cost_tensors.append(t)
        return torch.cat(cost_tensors, dim=1)

