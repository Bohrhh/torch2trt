import os
import cv2
import torch
import numpy as np
from torchvision import models
from torch2trt import torch2trt
from torch.utils.data import Dataset, DataLoader

class ImagenetSet(Dataset):

    """
    data folder organization
    ${data}
    |-- val/

    """

    def __init__(self, img_folder):
        super(Dataset, self).__init__()
        self.img_folder = img_folder
        self.images = os.listdir(img_folder)

    def __getitem__(self, index):
        img = os.path.join(self.img_folder, self.images[index])
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = (img/255.-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        img = img.transpose(2,0,1)[None].astype('float32')
        return img

    def __len__(self):
        return 100

def main():
    model = models.resnet50(pretrained=True)
    model = model.eval().to('cuda')
    x     = torch.rand(1,3,224,224).to('cuda')

    dataset = ImagenetSet('/home/kmlee/DATA/imagenet/val')

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )

    modeltrt = torch2trt(model, [x], int8_mode=True, int8_calib_dataset=dataloader)
    return modeltrt
