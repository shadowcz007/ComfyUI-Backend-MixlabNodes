import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import math
import folder_paths
from contextlib import nullcontext
from tqdm import tqdm
from nodes import MAX_RESOLUTION
import comfy.latent_formats
import comfy.model_management as mm
from comfy.utils import ProgressBar, unet_to_diffusers, load_torch_file
from comfy.model_base import BaseModel
from torchvision.transforms import Resize, CenterCrop
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFilter

# Utility functions from mtb nodes: https://github.com/melMass/comfy_mtb
# https://github.com/kijai/ComfyUI-KJNodes/blob/main/utility/utility.py
def pil2tensor(image):
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def bbox_to_region(bbox, target_size=None):
    bbox = bbox_check(bbox, target_size)
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

def bbox_check(bbox, target_size=None):
    if not target_size:
        return bbox

    new_bbox = (
        bbox[0],
        bbox[1],
        min(target_size[0] - bbox[0], bbox[2]),
        min(target_size[1] - bbox[1], bbox[3]),
    )
    return new_bbox


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
    
class MuseModelConfig:
    def __init__(self):
        unet_dtype = mm.unet_dtype()
        self.unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': unet_dtype, 'in_channels': 8, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 384, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}
        self.latent_format = comfy.latent_formats.SD15
        self.manual_cast_dtype = None
        self.sampling_settings = {}

class UNETLoader_MuseTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "MuseTalk"

    def load_unet(self):
        
        model_path = os.path.join(folder_paths.models_dir,'musetalk')
        
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="TMElyralab/MuseTalk", local_dir=model_path, local_dir_use_symlinks=False)

        unet_weight_path = os.path.join(model_path,"pytorch_model.bin") 
        
        sd = load_torch_file(unet_weight_path)
     
        model_config = MuseModelConfig()
        diffusers_keys = unet_to_diffusers(model_config.unet_config)
        
        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)

        model = BaseModel(model_config)
        model.diffusion_model.load_state_dict(new_sd, strict=False)
        return (model,)

class muse_talk_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "whisper_features" : ("WHISPERFEAT",),
            "images": ("IMAGE",),
            "masked_images": ("IMAGE",),
            "batch_size": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
            "delay_frame": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",  )
    RETURN_NAMES = ("image",  )
    FUNCTION = "process"
    CATEGORY = "MuseTalk"

    def process(self, model, vae, whisper_features, images, masked_images, batch_size, delay_frame):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = mm.unet_dtype()
        vae_scale_factor = 0.18215
        mm.unload_all_models()
        mm.soft_empty_cache()
        
        images = images.to(dtype).to(device)
        masked_images = masked_images.to(dtype).to(device)      

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            timesteps = torch.tensor([0], device=device)
            vae.first_stage_model.to(device)
            input_latent_list = []
            for image, masked_image in zip(images, masked_images):
                latent = vae.encode(image.unsqueeze(0)).to(dtype).to(device) * vae_scale_factor
                masked_latents = vae.encode(masked_image.unsqueeze(0)).to(dtype).to(device) * vae_scale_factor

                latent_model_input = torch.cat([masked_latents, latent], dim=1)
                input_latent_list.append(latent_model_input)

            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            video_num = len(whisper_features)
            gen = self.datagen(whisper_features, input_latent_list_cycle, batch_size, delay_frame)
            
            total=int(np.ceil(float(video_num)/batch_size))
            
            out_frame_list = []
            
            pbar = ProgressBar(total)
            print(total)
            model.diffusion_model.to(device)
            for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=total)):
                
                tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
                audio_feature_batch = torch.stack(tensor_list).to(device) # torch, B, 5*N,384
                audio_feature_batch = PositionalEncoding(d_model=384)(audio_feature_batch)
              
                pred_latents = model.diffusion_model(latent_batch, timesteps, context=audio_feature_batch)

                pred_latents = (1 / vae_scale_factor) * pred_latents
                decoded = vae.decode(pred_latents)
                
                for frame in decoded:
                    out_frame_list.append(frame)
                pbar.update(1)

            out = torch.stack(out_frame_list, dim=0).float().cpu()
        model.diffusion_model.to(offload_device)
        vae.first_stage_model.to(offload_device)
        return (out,)
    
    def datagen(self, whisper_chunks,vae_encode_latents,batch_size,delay_frame):
        whisper_batch, latent_batch = [], []
        for i, w in enumerate(whisper_chunks):
            idx = (i+delay_frame)%len(vae_encode_latents)
            latent = vae_encode_latents[idx]
            whisper_batch.append(w)
            latent_batch.append(latent)

            if len(latent_batch) >= batch_size:
                whisper_batch = np.asarray(whisper_batch)
                latent_batch = torch.cat(latent_batch, dim=0)
                yield whisper_batch, latent_batch
                whisper_batch, latent_batch = [], []

        # the last batch may smaller than batch size
        if len(latent_batch) > 0:
            whisper_batch = np.asarray(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)

            yield whisper_batch, latent_batch

class audio_file_to_audio_tensor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_file_path": ("STRING",  {"forceInput": True}),
            "target_sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
            "target_channels": ("INT", {"default": 1, "min": 1, "max": 2}),
             },
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_file_path, target_sample_rate, target_channels):
        
        audio_tensor, sample_rate = torchaudio.load(audio_file_path)
        assert audio_tensor.shape[0] in [1, 2], "Audio must be mono or stereo."
        if target_channels == 1:
            audio_tensor = audio_tensor.mean(0, keepdim=True)
        elif target_channels == 2:
            *shape, _, length = audio_tensor.shape
            audio_tensor = audio_tensor.expand(*shape, target_channels, length)
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.expand(target_channels, -1)
        resampled_audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)
        audio_dur = audio_tensor.shape[1] / target_sample_rate
        
        return (resampled_audio_tensor, audio_dur,)

class whisper_to_features:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "audio_tensor" : ("VCAUDIOTENSOR",),
                "fps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("WHISPERFEAT", "INT",)
    RETURN_NAMES = ("whisper_chunks", "frame_count",)
    FUNCTION = "whispertranscribe"
    CATEGORY = "VoiceCraft"

    def whispertranscribe(self, audio_tensor, fps):
        from .musetalk.whisper.model import Whisper, ModelDimensions
        device = mm.get_torch_device()

        model_path = os.path.join(folder_paths.models_dir,'whisper',"tiny.pt")
        
        if not os.path.exists(model_path):
            print(f"Downloading whisper tiny model (72MB) to {model_path}")
            import requests
            url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
            response = requests.get(url)
            if response.status_code == 200:
                with open(model_path, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Failed to download {url} to {model_path}, status code: {response.status_code}")
        whisper_sd = torch.load(model_path, map_location=device)
        dims = ModelDimensions(**whisper_sd["dims"])
        model = Whisper(dims)
        model.load_state_dict(whisper_sd["model_state_dict"])
        del whisper_sd
        result = model.transcribe(audio_tensor.squeeze(0))
        
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings']
            encoder_embeddings = encoder_embeddings.transpose(0,2,1,3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb['start'])
            end_idx = int(emb['end'])
            emb_end_idx = int((end_idx - start_idx)/2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        whisper_feature = np.concatenate(embed_list, axis=0)

        audio_feat_length = [2,2]
        whisper_chunks = []
        whisper_idx_multiplier = 50./fps 
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")
        while 1:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature,selected_idx = self.get_sliced_feature(feature_array= whisper_feature,vid_idx = i,audio_feat_length=audio_feat_length,fps=fps)
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx>len(whisper_feature):
                break
        print(f"Whisper chunks: {len(whisper_chunks)}")
        return (whisper_chunks, len(whisper_chunks),)
    
    def get_sliced_feature(self,feature_array, vid_idx, audio_feat_length= [2,2],fps = 25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx*50/fps) 
        left_idx = center_idx-audio_feat_length[0]*2
        right_idx = center_idx + (audio_feat_length[1]+1)*2
        
        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx
    

class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100,
                    "step": 0.1
                }),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
# GrowMaskWithBlur
- mask: Input mask or mask batch
- expand: Expand or contract mask or mask batch by a given amount
- incremental_expandrate: increase expand rate by a given amount per frame
- tapered_corners: use tapered corners
- flip_input: flip input mask
- blur_radius: value higher than 0 will blur the mask
- lerp_alpha: alpha value for interpolation between frames
- decay_factor: decay value for interpolation between frames
- fill_holes: fill holes in the mask (slow)"""
    
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy()
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)
        
class BatchCropFromMask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_size_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "bbox_smooth_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "BBOX",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "original_images",
        "cropped_images",
        "bboxes",
        "width",
        "height",
    )
    FUNCTION = "crop"
    CATEGORY = "KJNodes/masking"

    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if alpha == 0:
            return prev_bbox_size
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if alpha == 0:
            return prev_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1])
        )

    def crop(self, original_images, masks, crop_size_mult, bbox_smooth_alpha):
 
        bounding_boxes = []
        cropped_images = []

        self.max_bbox_width = 0
        self.max_bbox_height = 0

        # First, calculate the maximum bounding box size across all masks
        curr_max_bbox_width = 0
        curr_max_bbox_height = 0
        for mask in masks:
            _mask = tensor2pil(mask)[0]
            non_zero_indices = np.nonzero(np.array(_mask))
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            width = max_x - min_x
            height = max_y - min_y
            curr_max_bbox_width = max(curr_max_bbox_width, width)
            curr_max_bbox_height = max(curr_max_bbox_height, height)

        # Smooth the changes in the bounding box size
        self.max_bbox_width = self.smooth_bbox_size(self.max_bbox_width, curr_max_bbox_width, bbox_smooth_alpha)
        self.max_bbox_height = self.smooth_bbox_size(self.max_bbox_height, curr_max_bbox_height, bbox_smooth_alpha)

        # Apply the crop size multiplier
        self.max_bbox_width = round(self.max_bbox_width * crop_size_mult)
        self.max_bbox_height = round(self.max_bbox_height * crop_size_mult)
        bbox_aspect_ratio = self.max_bbox_width / self.max_bbox_height

        # Then, for each mask and corresponding image...
        for i, (mask, img) in enumerate(zip(masks, original_images)):
            _mask = tensor2pil(mask)[0]
            non_zero_indices = np.nonzero(np.array(_mask))
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            
            # Calculate center of bounding box
            center_x = np.mean(non_zero_indices[1])
            center_y = np.mean(non_zero_indices[0])
            curr_center = (round(center_x), round(center_y))

            # If this is the first frame, initialize prev_center with curr_center
            if not hasattr(self, 'prev_center'):
                self.prev_center = curr_center

            # Smooth the changes in the center coordinates from the second frame onwards
            if i > 0:
                center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
            else:
                center = curr_center

            # Update prev_center for the next frame
            self.prev_center = center

            # Create bounding box using max_bbox_width and max_bbox_height
            half_box_width = round(self.max_bbox_width / 2)
            half_box_height = round(self.max_bbox_height / 2)
            min_x = max(0, center[0] - half_box_width)
            max_x = min(img.shape[1], center[0] + half_box_width)
            min_y = max(0, center[1] - half_box_height)
            max_y = min(img.shape[0], center[1] + half_box_height)

            # Append bounding box coordinates
            bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

            # Crop the image from the bounding box
            cropped_img = img[min_y:max_y, min_x:max_x, :]
            
            # Calculate the new dimensions while maintaining the aspect ratio
            new_height = min(cropped_img.shape[0], self.max_bbox_height)
            new_width = round(new_height * bbox_aspect_ratio)

            # Resize the image
            resize_transform = Resize((new_height, new_width))
            resized_img = resize_transform(cropped_img.permute(2, 0, 1))

            # Perform the center crop to the desired size
            crop_transform = CenterCrop((self.max_bbox_height, self.max_bbox_width)) # swap the order here if necessary
            cropped_resized_img = crop_transform(resized_img)

            cropped_images.append(cropped_resized_img.permute(1, 2, 0))

        cropped_out = torch.stack(cropped_images, dim=0)
        
        return (original_images, cropped_out, bounding_boxes, self.max_bbox_width, self.max_bbox_height, )

class BatchUncrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "bboxes": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "border_top": ("BOOLEAN", {"default": True}),
                "border_bottom": ("BOOLEAN", {"default": True}),
                "border_left": ("BOOLEAN", {"default": True}),
                "border_right": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "uncrop"

    CATEGORY = "KJNodes/masking"

    def uncrop(self, original_images, cropped_images, bboxes, border_blending, crop_rescale, border_top, border_bottom, border_left, border_right):
        def inset_border(image, border_width, border_color, border_top, border_bottom, border_left, border_right):
            draw = ImageDraw.Draw(image)
            width, height = image.size
            if border_top:
                draw.rectangle((0, 0, width, border_width), fill=border_color)
            if border_bottom:
                draw.rectangle((0, height - border_width, width, height), fill=border_color)
            if border_left:
                draw.rectangle((0, 0, border_width, height), fill=border_color)
            if border_right:
                draw.rectangle((width - border_width, 0, width, height), fill=border_color)
            return image

        if len(original_images) != len(cropped_images):
            raise ValueError(f"The number of original_images ({len(original_images)}) and cropped_images ({len(cropped_images)}) should be the same")

        # Ensure there are enough bboxes, but drop the excess if there are more bboxes than images
        if len(bboxes) > len(original_images):
            print(f"Warning: Dropping excess bounding boxes. Expected {len(original_images)}, but got {len(bboxes)}")
            bboxes = bboxes[:len(original_images)]
        elif len(bboxes) < len(original_images):
            raise ValueError("There should be at least as many bboxes as there are original and cropped images")

        input_images = tensor2pil(original_images)
        crop_imgs = tensor2pil(cropped_images)
        
        out_images = []
        for i in range(len(input_images)):
            img = input_images[i]
            crop = crop_imgs[i]
            bbox = bboxes[i]
            
            # uncrop the image based on the bounding box
            bb_x, bb_y, bb_width, bb_height = bbox

            paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
            
            # scale factors
            scale_x = crop_rescale
            scale_y = crop_rescale

            # scaled paste_region
            paste_region = (round(paste_region[0]*scale_x), round(paste_region[1]*scale_y), round(paste_region[2]*scale_x), round(paste_region[3]*scale_y))

            # rescale the crop image to fit the paste_region
            crop = crop.resize((round(paste_region[2]-paste_region[0]), round(paste_region[3]-paste_region[1])))
            crop_img = crop.convert("RGB")
   
            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)

            blend = img.convert("RGBA")
            mask = Image.new("L", img.size, 0)

            mask_block = Image.new("L", (paste_region[2]-paste_region[0], paste_region[3]-paste_region[1]), 255)
            mask_block = inset_border(mask_block, round(blend_ratio / 2), (0), border_top, border_bottom, border_left, border_right)
                      
            mask.paste(mask_block, paste_region)
            blend.paste(crop_img, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            blend.putalpha(mask)
            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)

NODE_CLASS_MAPPINGS = {
    "whisper_to_features": whisper_to_features,
    "audio_file_to_audio_tensor": audio_file_to_audio_tensor,
    "muse_talk_sampler": muse_talk_sampler,
    "UNETLoader_MuseTalk": UNETLoader_MuseTalk,
    "GrowMaskWithBlur":GrowMaskWithBlur,
    "BatchCropFromMask":BatchCropFromMask,
    "BatchUncrop":BatchUncrop
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "whisper_to_features": "Whisper To Features",
    "audio_file_to_audio_tensor": "Audio File To Audio Tensor",
    "muse_talk_sampler": "MuseTalk Sampler",
    "UNETLoader_MuseTalk": "UNETLoader_MuseTalk"
}
