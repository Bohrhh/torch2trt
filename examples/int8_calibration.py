import os
import cv2
import torch
import numpy as np
# from torchvision import models
from torch2trt import torch2trt
from torch.utils.data import Dataset, DataLoader
import models.superpoint_lite as sl

class ImagenetSet(Dataset):

    """
    data folder organization
    ${data}
    |-- val/

    """

    def __init__(self, img_folder):
        super(ImagenetSet, self).__init__()
        self.img_folder = img_folder
        self.images = os.listdir(img_folder)

    def __getitem__(self, index):
        img = os.path.join(self.img_folder, self.images[index])
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 320))
        # img = (img/255.-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        img = img[None, None].astype('float32')
        return img

    def __len__(self):
        return 100

def main():
    # model = models.resnet50(pretrained=True)
    model = sl.SuperPointLite()
    model.load_state_dict(torch.load('models/superpoint_lite.pth')['model'])
    model = model.eval().to('cuda')
    x     = torch.rand(1,1,320,512).to('cuda')

    dataset = ImagenetSet('/home/kmlee/DATA/mscoco_2014/val2014')

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )

    modeltrt = torch2trt(model, [x], int8_mode=True, int8_calib_dataset=dataloader)
    return modeltrt