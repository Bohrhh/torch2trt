import torch
import torchvision
from .utils import add_module_test


# ========================================================================
# classification

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32]})
def alexnet():
    return torchvision.models.alexnet(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def squeezenet1_0():
    return torchvision.models.squeezenet1_0(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def squeezenet1_1():
    return torchvision.models.squeezenet1_1(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def resnet18():
    return torchvision.models.resnet18(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def resnet34():
    return torchvision.models.resnet34(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def resnet50():
    return torchvision.models.resnet50(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def resnet101():
    return torchvision.models.resnet101(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def resnet152():
    return torchvision.models.resnet152(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def densenet121():
    return torchvision.models.densenet121(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def densenet169():
    return torchvision.models.densenet169(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def densenet201():
    return torchvision.models.densenet201(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def densenet161():
    return torchvision.models.densenet161(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32]})
def vgg11():
    return torchvision.models.vgg11(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32]})
def vgg13_bn():
    return torchvision.models.vgg13_bn(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def mobilenet_v2():
    return torchvision.models.mobilenet_v2(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
# @add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32]})
def shufflenet_v2_x0_5():
    return torchvision.models.shufflenet_v2_x0_5(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
# @add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32]})
def shufflenet_v2_x1_0():
    return torchvision.models.shufflenet_v2_x1_0(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def mnasnet0_5():
    return torchvision.models.mnasnet0_5(pretrained=True)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def mnasnet1_0():
    return torchvision.models.mnasnet1_0(pretrained=True)


# ========================================================================
# segmentation

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        logits = self.model(x)['out']
        seg = torch.argmax(logits, dim=1, keepdim=True)
        return seg
    
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def deeplabv3_resnet50():
    bb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model = ModelWrapper(bb)
    return model

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def deeplabv3_resnet101():
    bb = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = ModelWrapper(bb)
    return model

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def fcn_resnet50():
    bb = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model = ModelWrapper(bb)
    return model

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, dynamic_axes={0:[1,32], 2:[128,256], 3:[128,256]})
def fcn_resnet101():
    bb = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
    model = ModelWrapper(bb)
    return model