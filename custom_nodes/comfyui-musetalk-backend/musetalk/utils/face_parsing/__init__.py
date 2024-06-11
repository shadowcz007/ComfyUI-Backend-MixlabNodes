import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
import torchvision.transforms as transforms

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MuseVCheckPointDir = os.path.join(
    diffusers_path, "TMElyralab/MuseTalk"
)


class FaceParsing():
    def __init__(self):
        self.net = self.model_init()
        self.preprocess = self.image_preprocess()

    def model_init(self, 
                   resnet_path=f'{MuseVCheckPointDir}/face-parse-bisent/resnet18-5c106cde.pth', 
                   model_pth=f'{MuseVCheckPointDir}/face-parse-bisent/79999_iter.pth'):
        net = BiSeNet(resnet_path)
        net.cuda()
        net.load_state_dict(torch.load(model_pth))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512)):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            img = torch.unsqueeze(img, 0).cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing[np.where(parsing>13)] = 0
            parsing[np.where(parsing>=1)] = 255
        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
    
