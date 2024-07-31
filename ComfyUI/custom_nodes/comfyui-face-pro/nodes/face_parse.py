import os
import numpy as np
import folder_paths
import torch
import torch.nn as nn
import matplotlib.cm as cm
import torchvision.transforms as T


def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)


face_parsing_path = get_model_dir("face_parsing")

from transformers import SegformerImageProcessor
from transformers import AutoModelForSemanticSegmentation


def parse_mask(
            result: torch.Tensor,
            background: bool,
            skin: bool,
            nose: bool,
            eyeglasses: bool,
            right_eye: bool,
            left_eye: bool,
            right_eyebrow: bool,
            left_eyebrow: bool,
            right_ear: bool,
            left_ear: bool,
            mouth_area_between_lips: bool,
            upper_lip: bool,
            lower_lip: bool,
            hair: bool,
            hat: bool,
            earring: bool,
            necklace: bool,
            neck: bool,
            clothing: bool):
        
    masks = []
    for item in result:
        mask = torch.zeros(item.shape, dtype=torch.uint8)
        if (background):
            mask = mask | torch.where(item == 0, 1, 0)
        if (skin):
            mask = mask | torch.where(item == 1, 1, 0)
        if (nose):
            mask = mask | torch.where(item == 2, 1, 0)    
        if (eyeglasses):
            mask = mask | torch.where(item == 3, 1, 0)  
        if (right_eye):
            mask = mask | torch.where(item == 4, 1, 0) 
        if (left_eye):
            mask = mask | torch.where(item == 5, 1, 0) 
        if (right_eyebrow):
            mask = mask | torch.where(item == 6, 1, 0) 
        if (left_eyebrow):
            mask = mask | torch.where(item == 7, 1, 0) 
        if (right_ear):
            mask = mask | torch.where(item == 8, 1, 0) 
        if (left_ear):
            mask = mask | torch.where(item == 9, 1, 0) 
        if (mouth_area_between_lips):
            mask = mask | torch.where(item == 10, 1, 0) 
        if (upper_lip):
            mask = mask | torch.where(item == 11, 1, 0) 
        if (lower_lip):
            mask = mask | torch.where(item == 12, 1, 0)   
        if (hair):
            mask = mask | torch.where(item == 13, 1, 0) 
        if (hat):
            mask = mask | torch.where(item == 14, 1, 0) 
        if (earring):
            mask = mask | torch.where(item == 15, 1, 0) 
        if (necklace):
            mask = mask | torch.where(item == 16, 1, 0) 
        if (neck):
            mask = mask | torch.where(item == 17, 1, 0)   
        if (clothing):
            mask = mask | torch.where(item == 18, 1, 0)         
        masks.append(mask.float())
    final = torch.cat(masks, dim=0).unsqueeze(0)
    return final


class FaceParse:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACEPARSE_",)
    RETURN_NAMES = ("debug_image","result",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Face"

    def run(self,image,debug ):

        processor = SegformerImageProcessor.from_pretrained(face_parsing_path)
        
        model = AutoModelForSemanticSegmentation.from_pretrained(face_parsing_path)
        
        images = []
        results = []
        transform = T.ToPILImage()
        colormap = cm.get_cmap('viridis', 19)

        for item in image:
            size = item.shape[:2]
            inputs = processor(images=transform(item.permute(2, 0, 1)), return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False)
            
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_np = pred_seg.detach().numpy().astype(np.uint8)
            results.append(torch.tensor(pred_seg_np))
            
            if debug:
                colored = colormap(pred_seg_np)
                colored_sliced = colored[:,:,:3] # type: ignore
                images.append(torch.tensor(colored_sliced))

        del processor
        del model

        d_img=None
        if debug:
            d_img=torch.cat(images, dim=0).unsqueeze(0)
        return (d_img, torch.cat(results, dim=0).unsqueeze(0),)



    
class FaceParsingResults:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_parse_result": ("FACEPARSE_", {}),
                "background": ("BOOLEAN", {"default": False}),
                "skin": ("BOOLEAN", {"default": True}),
                "nose": ("BOOLEAN", {"default": True}),
                "eyeglasses": ("BOOLEAN", {"default": True}),
                "right_eye": ("BOOLEAN", {"default": True}),
                "left_eye": ("BOOLEAN", {"default": True}),
                "right_eyebrow": ("BOOLEAN", {"default": True}),
                "left_eyebrow": ("BOOLEAN", {"default": True}),
                "right_ear": ("BOOLEAN", {"default": True}),
                "left_ear": ("BOOLEAN", {"default": True}),
                "mouth_area_between_lips": ("BOOLEAN", {"default": True}),
                "upper_lip": ("BOOLEAN", {"default": True}),
                "lower_lip": ("BOOLEAN", {"default": True}),
                "hair": ("BOOLEAN", {"default": True}),
                "hat": ("BOOLEAN", {"default": True}),
                "earring": ("BOOLEAN", {"default": True}),
                "necklace": ("BOOLEAN", {"default": True}),
                "neck": ("BOOLEAN", {"default": True}),
                "clothing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    
    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Face"

    def run(
            self, 
            face_parse_result: torch.Tensor,
            background: bool,
            skin: bool,
            nose: bool,
            eyeglasses: bool,
            right_eye: bool,
            left_eye: bool,
            right_eyebrow: bool,
            left_eyebrow: bool,
            right_ear: bool,
            left_ear: bool,
            mouth_area_between_lips: bool,
            upper_lip: bool,
            lower_lip: bool,
            hair: bool,
            hat: bool,
            earring: bool,
            necklace: bool,
            neck: bool,
            clothing: bool):
        
        mask=parse_mask(face_parse_result,
                        background ,
            skin ,
            nose ,
            eyeglasses ,
            right_eye,
            left_eye ,
            right_eyebrow ,
            left_eyebrow ,
            right_ear ,
            left_ear ,
            mouth_area_between_lips ,
            upper_lip ,
            lower_lip ,
            hair ,
            hat ,
            earring,
            necklace,
            neck ,
            clothing 
            )

        return (mask,)
